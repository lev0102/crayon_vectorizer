#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

try:
    from websocket import create_connection
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: websocket-client\n"
        "Install with: pip install websocket-client watchdog"
    ) from exc


DEFAULT_CONFIG = {
    "websocket": {
        "url": "ws://127.0.0.1:8080",
        "connect_on_start": True,
        "reconnect_on_send_failure": True
    },
    "watcher": {
        "recursive": False,
        "poll_interval_seconds": 1.0,
        "stable_check_interval_seconds": 0.2,
        "stable_checks_required": 3,
        "default_debounce_seconds": 0.5,
        "log_level": "INFO"
    },
    "stage1_source_jpg_watch": {
        "enabled": True,
        "folder": r"C:\\Pipeline\\InputA",
        "file_regex": r"^image\d+\.jpg$",
        "case_sensitive": False,
        "event_type": "source_jpg_ready",
        "queue": {
            "enabled": False,
            "cooldown_seconds": 0.0,
            "deduplicate_by_full_path": True
        }
    },
    "stage2_processed_jpg_watch": {
        "enabled": True,
        "folder": r"C:\\Pipeline\\InputB",
        "file_regex": r"^image\d+\.jpg$",
        "case_sensitive": False,
        "run_pipeline": {
            "enabled": True,
            "python_executable": "",
            "script_path": "run_locked_palette_pipeline.py",
            "config_path": "locked_palette_pipeline_config.json",
            "outdir": r"C:\\Pipeline\\Output",
            "extra_args": []
        }
    },
    "stage3_spawn_json_watch": {
        "enabled": True,
        "folder": r"C:\\Pipeline\\Output",
        "file_regex": r"^output_output_.*_vector_spawn_points\.json$",
        "case_sensitive": False,
        "event_type": "spawn_json_ready",
        "queue": {
            "enabled": True,
            "cooldown_seconds": 30.0,
            "deduplicate_by_full_path": True
        }
    }
}


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Path) -> dict:
    cfg = DEFAULT_CONFIG
    if path.exists():
        cfg = deep_merge(cfg, load_json(path))
    return cfg


class WebSocketSender:
    def __init__(self, url: str, reconnect_on_send_failure: bool = True):
        self.url = url
        self.reconnect_on_send_failure = reconnect_on_send_failure
        self.ws = None
        self.lock = threading.Lock()

    def connect(self) -> None:
        with self.lock:
            if self.ws is None:
                self.ws = create_connection(self.url)
                logging.info("Connected to WebSocket: %s", self.url)

    def close(self) -> None:
        with self.lock:
            if self.ws is not None:
                try:
                    self.ws.close()
                except Exception:
                    pass
                self.ws = None

    def send_json(self, payload: dict) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        try:
            with self.lock:
                if self.ws is None:
                    self.ws = create_connection(self.url)
                    logging.info("Connected to WebSocket: %s", self.url)
                self.ws.send(message)
            logging.info("Sent -> %s", message)
        except Exception as e:
            logging.error("WebSocket send failed: %s", e)
            if self.reconnect_on_send_failure:
                with self.lock:
                    try:
                        if self.ws is not None:
                            self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                raise


def normalize_name(name: str, case_sensitive: bool) -> str:
    return name if case_sensitive else name.lower()


def make_exact_name_matcher(target_file_name: str, case_sensitive: bool) -> Callable[[Path], bool]:
    normalized_target = normalize_name(target_file_name, case_sensitive)

    def _match(path: Path) -> bool:
        return normalize_name(path.name, case_sensitive) == normalized_target

    return _match


def make_regex_name_matcher(pattern: str, case_sensitive: bool) -> Callable[[Path], bool]:
    flags = 0 if case_sensitive else re.IGNORECASE
    rx = re.compile(pattern, flags=flags)

    def _match(path: Path) -> bool:
        return rx.match(path.name) is not None

    return _match


def build_name_matcher(node: dict) -> Callable[[Path], bool]:
    case_sensitive = bool(node.get("case_sensitive", False))
    file_regex = str(node.get("file_regex", "")).strip()
    target_file_name = str(node.get("target_file_name", "")).strip()

    if file_regex:
        return make_regex_name_matcher(file_regex, case_sensitive)

    if target_file_name:
        return make_exact_name_matcher(target_file_name, case_sensitive)

    raise ValueError("Config must provide either 'file_regex' or 'target_file_name'.")


def wait_until_file_stable(path: Path, check_interval_seconds: float, stable_checks_required: int) -> bool:
    last_signature = None
    stable_count = 0

    while True:
        if not path.exists() or not path.is_file():
            return False

        try:
            stat = path.stat()
            signature = (stat.st_size, int(stat.st_mtime_ns))
        except OSError:
            return False

        if signature == last_signature:
            stable_count += 1
            if stable_count >= stable_checks_required:
                return True
        else:
            stable_count = 0
            last_signature = signature

        time.sleep(check_interval_seconds)


@dataclass
class QueueSettings:
    enabled: bool = False
    cooldown_seconds: float = 0.0
    deduplicate_by_full_path: bool = True


@dataclass
class WatchTarget:
    name: str
    folder: Path
    matcher: Callable[[Path], bool]
    debounce_seconds: float
    stable_check_interval_seconds: float
    stable_checks_required: int
    event_type: str | None = None
    queue: QueueSettings = field(default_factory=QueueSettings)


class DelayedQueueDispatcher:
    def __init__(self, target_name: str, queue_settings: QueueSettings, send_func: Callable[[dict], None]):
        self.target_name = target_name
        self.queue_settings = queue_settings
        self.send_func = send_func
        self.lock = threading.Lock()
        self.queue = []
        self.pending_by_key = {}
        self.timer = None
        self.cooldown_active = False

    def _payload_key(self, payload: dict) -> str:
        return str(payload.get("full_path", ""))

    def _flush(self) -> None:
        with self.lock:
            items = list(self.queue)
            self.queue.clear()
            self.pending_by_key.clear()
            self.timer = None
            self.cooldown_active = False

        if items:
            logging.info("[%s] Flushing %d queued item(s)", self.target_name, len(items))
        for payload in items:
            self.send_func(payload)

    def enqueue_or_send(self, payload: dict) -> None:
        if not self.queue_settings.enabled or self.queue_settings.cooldown_seconds <= 0.0:
            self.send_func(payload)
            return

        send_now = False
        with self.lock:
            if not self.cooldown_active:
                self.cooldown_active = True
                self.timer = threading.Timer(self.queue_settings.cooldown_seconds, self._flush)
                self.timer.daemon = True
                self.timer.start()
                logging.info(
                    "[%s] Queue cooldown started for %.2f sec; sending first item immediately",
                    self.target_name,
                    self.queue_settings.cooldown_seconds,
                )
                send_now = True
            else:
                if self.queue_settings.deduplicate_by_full_path:
                    key = self._payload_key(payload)
                    if key in self.pending_by_key:
                        self.pending_by_key[key] = payload
                        for i, existing in enumerate(self.queue):
                            if self._payload_key(existing) == key:
                                self.queue[i] = payload
                                break
                        logging.info("[%s] Updated queued item: %s", self.target_name, key)
                    else:
                        self.queue.append(payload)
                        self.pending_by_key[key] = payload
                        logging.info("[%s] Queued item: %s", self.target_name, key)
                else:
                    self.queue.append(payload)
                    logging.info("[%s] Queued item: %s", self.target_name, payload.get("full_path"))

        if send_now:
            self.send_func(payload)

    def shutdown(self) -> None:
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None


class PipelineRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.last_seen_time_by_path = {}

    def _debounced(self, path_str: str, debounce_seconds: float) -> bool:
        now = time.time()
        with self.lock:
            last = self.last_seen_time_by_path.get(path_str, 0.0)
            if now - last < debounce_seconds:
                return True
            self.last_seen_time_by_path[path_str] = now
        return False

    def run_for_input(self, input_path: Path, event_name: str, target_name: str) -> None:
        stage_cfg = self.cfg["stage2_processed_jpg_watch"]
        run_cfg = stage_cfg["run_pipeline"]
        if not bool(run_cfg.get("enabled", True)):
            return

        path_str = str(input_path.resolve())
        debounce_seconds = float(self.cfg["watcher"].get("default_debounce_seconds", 0.5))
        if self._debounced(path_str, debounce_seconds):
            return

        logging.info("[%s] Candidate matched for pipeline run: %s (%s)", target_name, path_str, event_name)

        if not wait_until_file_stable(
            input_path,
            check_interval_seconds=float(self.cfg["watcher"].get("stable_check_interval_seconds", 0.2)),
            stable_checks_required=int(self.cfg["watcher"].get("stable_checks_required", 3)),
        ):
            logging.warning("[%s] File disappeared before stabilizing: %s", target_name, path_str)
            return

        python_exe = str(run_cfg.get("python_executable", "")).strip() or sys.executable
        script_path = Path(str(run_cfg["script_path"])).expanduser().resolve()
        config_path = Path(str(run_cfg["config_path"])).expanduser().resolve()
        outdir = str(run_cfg.get("outdir", "")).strip()
        extra_args = [str(x) for x in run_cfg.get("extra_args", [])]

        cmd = [
            python_exe,
            str(script_path),
            "--input",
            path_str,
            "--config",
            str(config_path),
        ]
        if outdir:
            cmd.extend(["--outdir", outdir])
        cmd.extend(extra_args)

        logging.info("[%s] Running pipeline: %s", target_name, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            logging.info("[%s] Pipeline completed for: %s", target_name, path_str)
        except subprocess.CalledProcessError as e:
            logging.error("[%s] Pipeline failed with code %s for %s", target_name, e.returncode, path_str)
        except Exception as e:
            logging.error("[%s] Pipeline failed for %s: %s", target_name, path_str, e)


class WsForwardingHandler(FileSystemEventHandler):
    def __init__(self, target: WatchTarget, dispatcher: DelayedQueueDispatcher):
        super().__init__()
        self.target = target
        self.dispatcher = dispatcher
        self.last_seen_time_by_path = {}
        self.lock = threading.Lock()

    def _debounced(self, path_str: str) -> bool:
        now = time.time()
        with self.lock:
            last = self.last_seen_time_by_path.get(path_str, 0.0)
            if now - last < self.target.debounce_seconds:
                return True
            self.last_seen_time_by_path[path_str] = now
        return False

    def _maybe_emit(self, file_path: Path, event_name: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            return
        if not self.target.matcher(file_path):
            return

        path_str = str(file_path.resolve())
        if self._debounced(path_str):
            return

        logging.info("[%s] Candidate matched: %s (%s)", self.target.name, path_str, event_name)

        if not wait_until_file_stable(
            file_path,
            check_interval_seconds=self.target.stable_check_interval_seconds,
            stable_checks_required=self.target.stable_checks_required,
        ):
            logging.warning("[%s] File disappeared before stabilizing: %s", self.target.name, path_str)
            return

        payload = {
            "type": self.target.event_type,
            "full_path": path_str,
            "file_name": file_path.name,
            "event": event_name,
            "timestamp": time.time(),
            "queue_name": self.target.name,
        }
        self.dispatcher.enqueue_or_send(payload)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_emit(Path(event.src_path), "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_emit(Path(event.src_path), "modified")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_emit(Path(event.dest_path), "moved")


class PipelineTriggerHandler(FileSystemEventHandler):
    def __init__(self, target: WatchTarget, pipeline_runner: PipelineRunner):
        super().__init__()
        self.target = target
        self.pipeline_runner = pipeline_runner

    def _maybe_run(self, file_path: Path, event_name: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            return
        if not self.target.matcher(file_path):
            return
        self.pipeline_runner.run_for_input(file_path, event_name, self.target.name)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_run(Path(event.src_path), "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_run(Path(event.src_path), "modified")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_run(Path(event.dest_path), "moved")


def build_queue_settings(node: dict) -> QueueSettings:
    q = node.get("queue", {})
    return QueueSettings(
        enabled=bool(q.get("enabled", False)),
        cooldown_seconds=float(q.get("cooldown_seconds", 0.0)),
        deduplicate_by_full_path=bool(q.get("deduplicate_by_full_path", True)),
    )


def build_stage1_target(cfg: dict):
    watcher_cfg = cfg["watcher"]
    node = cfg.get("stage1_source_jpg_watch", {})
    if not node.get("enabled", True):
        return None
    return WatchTarget(
        name="stage1_source_jpg_watch",
        folder=Path(node["folder"]).expanduser(),
        matcher=build_name_matcher(node),
        debounce_seconds=float(watcher_cfg.get("default_debounce_seconds", 0.5)),
        stable_check_interval_seconds=float(watcher_cfg.get("stable_check_interval_seconds", 0.2)),
        stable_checks_required=int(watcher_cfg.get("stable_checks_required", 3)),
        event_type=str(node.get("event_type", "source_jpg_ready")),
        queue=build_queue_settings(node),
    )


def build_stage2_target(cfg: dict):
    watcher_cfg = cfg["watcher"]
    node = cfg.get("stage2_processed_jpg_watch", {})
    if not node.get("enabled", True):
        return None
    return WatchTarget(
        name="stage2_processed_jpg_watch",
        folder=Path(node["folder"]).expanduser(),
        matcher=build_name_matcher(node),
        debounce_seconds=float(watcher_cfg.get("default_debounce_seconds", 0.5)),
        stable_check_interval_seconds=float(watcher_cfg.get("stable_check_interval_seconds", 0.2)),
        stable_checks_required=int(watcher_cfg.get("stable_checks_required", 3)),
    )


def build_stage3_target(cfg: dict):
    watcher_cfg = cfg["watcher"]
    node = cfg.get("stage3_spawn_json_watch", {})
    if not node.get("enabled", True):
        return None
    return WatchTarget(
        name="stage3_spawn_json_watch",
        folder=Path(node["folder"]).expanduser(),
        matcher=build_name_matcher(node),
        debounce_seconds=float(watcher_cfg.get("default_debounce_seconds", 0.5)),
        stable_check_interval_seconds=float(watcher_cfg.get("stable_check_interval_seconds", 0.2)),
        stable_checks_required=int(watcher_cfg.get("stable_checks_required", 3)),
        event_type=str(node.get("event_type", "spawn_json_ready")),
        queue=build_queue_settings(node),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="3-stage file watcher for Unreal + Python pipeline.")
    ap.add_argument("--config", default="three_stage_watcher_config.json", help="Path to config JSON.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    log_level_name = str(cfg["watcher"].get("log_level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    ws_cfg = cfg["websocket"]
    sender = WebSocketSender(
        url=str(ws_cfg["url"]),
        reconnect_on_send_failure=bool(ws_cfg.get("reconnect_on_send_failure", True)),
    )

    if bool(ws_cfg.get("connect_on_start", True)):
        try:
            sender.connect()
        except Exception as e:
            logging.warning("Initial WebSocket connect failed: %s", e)

    recursive = bool(cfg["watcher"].get("recursive", False))
    poll_interval = float(cfg["watcher"].get("poll_interval_seconds", 1.0))

    stage1 = build_stage1_target(cfg)
    stage2 = build_stage2_target(cfg)
    stage3 = build_stage3_target(cfg)

    observers = []
    dispatchers = {}

    try:
        if stage1 is not None:
            stage1.folder.mkdir(parents=True, exist_ok=True)
            dispatcher1 = DelayedQueueDispatcher(stage1.name, stage1.queue, sender.send_json)
            dispatchers[stage1.name] = dispatcher1
            handler1 = WsForwardingHandler(stage1, dispatcher1)
            observer1 = Observer(timeout=poll_interval)
            observer1.schedule(handler1, str(stage1.folder), recursive=recursive)
            observer1.start()
            observers.append(observer1)
            logging.info("Watching [%s] folder: %s", stage1.name, str(stage1.folder.resolve()))

        if stage2 is not None:
            stage2.folder.mkdir(parents=True, exist_ok=True)
            pipeline_runner = PipelineRunner(cfg)
            handler2 = PipelineTriggerHandler(stage2, pipeline_runner)
            observer2 = Observer(timeout=poll_interval)
            observer2.schedule(handler2, str(stage2.folder), recursive=recursive)
            observer2.start()
            observers.append(observer2)
            logging.info("Watching [%s] folder: %s", stage2.name, str(stage2.folder.resolve()))

        if stage3 is not None:
            stage3.folder.mkdir(parents=True, exist_ok=True)
            dispatcher3 = DelayedQueueDispatcher(stage3.name, stage3.queue, sender.send_json)
            dispatchers[stage3.name] = dispatcher3
            handler3 = WsForwardingHandler(stage3, dispatcher3)
            observer3 = Observer(timeout=poll_interval)
            observer3.schedule(handler3, str(stage3.folder), recursive=recursive)
            observer3.start()
            observers.append(observer3)
            logging.info("Watching [%s] folder: %s", stage3.name, str(stage3.folder.resolve()))
            if stage3.queue.enabled:
                logging.info(
                    "[%s] Queue enabled, cooldown=%.2f sec, dedupe=%s",
                    stage3.name,
                    stage3.queue.cooldown_seconds,
                    stage3.queue.deduplicate_by_full_path,
                )

        if not observers:
            raise SystemExit("No stages enabled in config.")

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
    finally:
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()
        for dispatcher in dispatchers.values():
            dispatcher.shutdown()
        sender.close()


if __name__ == "__main__":
    main()
