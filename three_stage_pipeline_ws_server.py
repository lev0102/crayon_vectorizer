# #!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
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
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: websockets\n"
        "Install with: pip install websockets watchdog"
    ) from exc


DEFAULT_CONFIG = {
    "websocket_server": {
        "host": "127.0.0.1",
        "port": 8080,
        "log_client_messages": False
    },
    "message_format": {
        "wrap_in_payload": True,
        "envelope_type": "pipeline_event"
    },
    "pipeline_cooldown": {
        "enabled": False,
        "cooldown_seconds": 0.0
    },
    "stage3_ack_gate": {
        "enabled": True,
        "ack_message_type": "stage2_ack",
        "match_by": "pipeline_key",
        "fallback_send_after_seconds": 0.0,
        "log_ack_messages": True
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
        "file_regex": r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.jpg$",
        "case_sensitive": False,
        "event_type": "source_jpg_ready",
        "allowed_events": ["created", "moved"],
        "queue": {
            "enabled": False,
            "cooldown_seconds": 0.0,
            "deduplicate_by_full_path": True
        }
    },
    "stage2_processed_jpg_watch": {
        "enabled": True,
        "folder": r"C:\\Pipeline\\InputB",
        "file_regex": r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.jpg$",
        "case_sensitive": False,
        "allowed_events": ["created", "moved"],
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
        "allowed_events": ["created", "moved"],
        "queue": {
            "enabled": False,
            "cooldown_seconds": 0.0,
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


def extract_pipeline_key(path: Path) -> str:
    """
    For the new naming scheme, use the timestamp stem directly as the pipeline key.
    Example: 2026-04-16_08-28-56.jpg -> 2026-04-16_08-28-56
    This keeps stage 1 / stage 2 / stage 3 / ACK matching stable as long as they
    all refer to the same timestamp-based filename or emitted pipeline_key.
    """
    return path.stem.lower()


def extract_ack_key_from_message(message_obj: dict, match_by: str) -> str | None:
    value = message_obj.get(match_by)
    if value is None:
        return None
    return str(value).strip().lower() or None


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
    allowed_events: set[str]
    event_type: str | None = None
    queue: QueueSettings = field(default_factory=QueueSettings)


class PipelineCooldownManager:
    """
    Cooldown only for full pipeline completion messages.
    Intended to gate Stage 3 sends, not Stage 1 or Stage 2.
    Keyed by image id if it can be extracted, otherwise by file stem.
    """
    def __init__(self, enabled: bool, cooldown_seconds: float):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.lock = threading.Lock()
        self.last_sent_by_key: dict[str, float] = {}

    def allow(self, pipeline_key: str) -> bool:
        if not self.enabled or self.cooldown_seconds <= 0.0:
            return True

        now = time.time()
        with self.lock:
            last = self.last_sent_by_key.get(pipeline_key, 0.0)
            if now - last < self.cooldown_seconds:
                return False
            self.last_sent_by_key[pipeline_key] = now
            return True


class WebSocketHub:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        log_client_messages: bool = False,
        ack_message_type: str = "stage2_ack",
        ack_match_by: str = "pipeline_key",
        ack_log_messages: bool = True,
    ):
        self.loop = loop
        self.log_client_messages = log_client_messages
        self.ack_message_type = ack_message_type
        self.ack_match_by = ack_match_by
        self.ack_log_messages = ack_log_messages
        self.clients: set[WebSocketServerProtocol] = set()
        self.clients_lock = threading.Lock()
        self.ack_gate = None

    def set_ack_gate(self, ack_gate) -> None:
        self.ack_gate = ack_gate

    async def register(self, ws: WebSocketServerProtocol) -> None:
        with self.clients_lock:
            self.clients.add(ws)
        logging.info("WebSocket client connected. clients=%d", len(self.clients))

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        with self.clients_lock:
            self.clients.discard(ws)
        logging.info("WebSocket client disconnected. clients=%d", len(self.clients))

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        await self.register(websocket)
        try:
            async for message in websocket:
                if self.log_client_messages:
                    logging.info("Client -> %s", message)

                try:
                    obj = json.loads(message)
                except Exception:
                    continue

                msg_type = str(obj.get("type", "")).strip()
                if msg_type == self.ack_message_type and self.ack_gate is not None:
                    ack_key = extract_ack_key_from_message(obj, self.ack_match_by)
                    if ack_key:
                        if self.ack_log_messages:
                            logging.info("ACK received type=%s %s=%s", msg_type, self.ack_match_by, ack_key)
                        self.ack_gate.note_ack(ack_key, raw_message=obj)
        except Exception as e:
            logging.warning("WebSocket client handler ended: %s", e)
        finally:
            await self.unregister(websocket)

    async def _broadcast(self, payload: dict) -> None:
        message = json.dumps(payload, ensure_ascii=False)
        with self.clients_lock:
            clients = list(self.clients)

        if not clients:
            logging.warning("No connected WebSocket clients. Dropping payload: %s", message)
            return

        dead = []
        for client in clients:
            try:
                await client.send(message)
            except Exception as e:
                logging.warning("Failed to send to client: %s", e)
                dead.append(client)

        if dead:
            with self.clients_lock:
                for client in dead:
                    self.clients.discard(client)

        logging.info("Broadcast -> %s", message)

    def send_json_threadsafe(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)


class Stage3AckGate:
    def __init__(
        self,
        enabled: bool,
        match_by: str,
        fallback_send_after_seconds: float,
        send_func: Callable[[dict], None],
    ):
        self.enabled = enabled
        self.match_by = match_by
        self.fallback_send_after_seconds = fallback_send_after_seconds
        self.send_func = send_func
        self.lock = threading.Lock()
        self.acked_keys: set[str] = set()
        self.pending_payloads: dict[str, dict] = {}
        self.pending_timers: dict[str, threading.Timer] = {}

    def _extract_key(self, payload: dict) -> str | None:
        value = payload.get(self.match_by)
        if value is None:
            return None
        return str(value).strip().lower() or None

    def _cancel_timer_locked(self, key: str) -> None:
        timer = self.pending_timers.pop(key, None)
        if timer is not None:
            timer.cancel()

    def _fallback_fire(self, key: str) -> None:
        payload = None
        with self.lock:
            payload = self.pending_payloads.pop(key, None)
            self.pending_timers.pop(key, None)
        if payload is not None:
            logging.info("[stage3_ack_gate] Fallback releasing pending payload for %s", key)
            self.send_func(payload)

    def note_ack(self, key: str, raw_message: dict | None = None) -> None:
        payload = None
        with self.lock:
            existing = self.pending_payloads.pop(key, None)
            self._cancel_timer_locked(key)
            if existing is not None:
                payload = existing
            else:
                self.acked_keys.add(key)

        if payload is not None:
            logging.info("[stage3_ack_gate] ACK matched pending payload for %s; sending now", key)
            self.send_func(payload)
        else:
            logging.info("[stage3_ack_gate] ACK stored early for %s", key)

    def enqueue_or_send(self, payload: dict) -> None:
        if not self.enabled:
            self.send_func(payload)
            return

        key = self._extract_key(payload)
        if not key:
            logging.warning("[stage3_ack_gate] Missing %s in payload; sending immediately", self.match_by)
            self.send_func(payload)
            return

        send_now = False
        with self.lock:
            if key in self.acked_keys:
                self.acked_keys.discard(key)
                send_now = True
            else:
                self.pending_payloads[key] = payload
                self._cancel_timer_locked(key)
                if self.fallback_send_after_seconds > 0.0:
                    timer = threading.Timer(self.fallback_send_after_seconds, self._fallback_fire, args=[key])
                    timer.daemon = True
                    timer.start()
                    self.pending_timers[key] = timer
                logging.info("[stage3_ack_gate] Holding stage3 payload until ACK for %s", key)

        if send_now:
            logging.info("[stage3_ack_gate] ACK already arrived for %s; sending now", key)
            self.send_func(payload)

    def shutdown(self) -> None:
        with self.lock:
            timers = list(self.pending_timers.values())
            self.pending_timers.clear()
            self.pending_payloads.clear()
            self.acked_keys.clear()
        for timer in timers:
            timer.cancel()


class MessageFormatter:

    def __init__(self, cfg: dict):
        self.wrap_in_payload = bool(cfg.get("message_format", {}).get("wrap_in_payload", True))
        self.envelope_type = str(cfg.get("message_format", {}).get("envelope_type", "pipeline_event"))

    def format(self, payload: dict) -> dict:
        if not self.wrap_in_payload:
            return payload
        return {
            "type": self.envelope_type,
            "payload": payload
        }


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
    def __init__(
        self,
        target: WatchTarget,
        dispatcher: DelayedQueueDispatcher,
        formatter: MessageFormatter,
        pipeline_cooldown: PipelineCooldownManager | None = None,
        apply_pipeline_cooldown: bool = False,
    ):
        super().__init__()
        self.target = target
        self.dispatcher = dispatcher
        self.formatter = formatter
        self.pipeline_cooldown = pipeline_cooldown
        self.apply_pipeline_cooldown = apply_pipeline_cooldown
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
        if event_name not in self.target.allowed_events:
            return
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

        pipeline_key = extract_pipeline_key(file_path)
        if self.apply_pipeline_cooldown and self.pipeline_cooldown is not None:
            if not self.pipeline_cooldown.allow(pipeline_key):
                logging.info(
                    "[%s] Suppressed by pipeline cooldown for key=%s path=%s",
                    self.target.name,
                    pipeline_key,
                    path_str,
                )
                return

        payload = {
            "stage_type": self.target.event_type,
            "full_path": path_str,
            "file_name": file_path.name,
            "event": event_name,
            "timestamp": time.time(),
            "queue_name": self.target.name,
            "pipeline_key": pipeline_key,
        }
        self.dispatcher.enqueue_or_send(self.formatter.format(payload))

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
        if event_name not in self.target.allowed_events:
            return
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


def build_allowed_events(node: dict) -> set[str]:
    return {str(x).strip().lower() for x in node.get("allowed_events", ["created", "modified", "moved"]) if str(x).strip()}


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
        allowed_events=build_allowed_events(node),
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
        allowed_events=build_allowed_events(node),
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
        allowed_events=build_allowed_events(node),
        event_type=str(node.get("event_type", "spawn_json_ready")),
        queue=build_queue_settings(node),
    )


async def main_async() -> None:
    ap = argparse.ArgumentParser(description="3-stage watcher with Python-hosted WebSocket server.")
    ap.add_argument("--config", default="three_stage_ws_server_v2_config.json", help="Path to config JSON.")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    log_level_name = str(cfg["watcher"].get("log_level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    loop = asyncio.get_running_loop()
    ws_cfg = cfg["websocket_server"]
    ack_gate_cfg = cfg.get("stage3_ack_gate", {})
    hub = WebSocketHub(
        loop=loop,
        log_client_messages=bool(ws_cfg.get("log_client_messages", False)),
        ack_message_type=str(ack_gate_cfg.get("ack_message_type", "stage2_ack")),
        ack_match_by=str(ack_gate_cfg.get("match_by", "pipeline_key")),
        ack_log_messages=bool(ack_gate_cfg.get("log_ack_messages", True)),
    )
    formatter = MessageFormatter(cfg)

    pipeline_cd_cfg = cfg.get("pipeline_cooldown", {})
    pipeline_cooldown = PipelineCooldownManager(
        enabled=bool(pipeline_cd_cfg.get("enabled", True)),
        cooldown_seconds=float(pipeline_cd_cfg.get("cooldown_seconds", 30.0)),
    )

    host = str(ws_cfg.get("host", "127.0.0.1"))
    port = int(ws_cfg.get("port", 8080))

    server = await websockets.serve(hub.handler, host, port)
    logging.info("WebSocket server listening on ws://%s:%d", host, port)

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
            dispatcher1 = DelayedQueueDispatcher(stage1.name, stage1.queue, hub.send_json_threadsafe)
            dispatchers[stage1.name] = dispatcher1
            handler1 = WsForwardingHandler(
                stage1,
                dispatcher1,
                formatter=formatter,
                pipeline_cooldown=None,
                apply_pipeline_cooldown=False,
            )
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
            dispatcher3 = DelayedQueueDispatcher(stage3.name, stage3.queue, hub.send_json_threadsafe)
            stage3_ack_gate = Stage3AckGate(
                enabled=bool(ack_gate_cfg.get("enabled", True)),
                match_by=str(ack_gate_cfg.get("match_by", "pipeline_key")),
                fallback_send_after_seconds=float(ack_gate_cfg.get("fallback_send_after_seconds", 0.0)),
                send_func=dispatcher3.enqueue_or_send,
            )
            hub.set_ack_gate(stage3_ack_gate)
            dispatchers[stage3.name] = stage3_ack_gate
            handler3 = WsForwardingHandler(
                stage3,
                dispatcher3,
                formatter=formatter,
                pipeline_cooldown=pipeline_cooldown,
                apply_pipeline_cooldown=bool(pipeline_cd_cfg.get("enabled", False)),
            )
            observer3 = Observer(timeout=poll_interval)
            observer3.schedule(handler3, str(stage3.folder), recursive=recursive)
            observer3.start()
            observers.append(observer3)
            logging.info("Watching [%s] folder: %s", stage3.name, str(stage3.folder.resolve()))

        if not observers:
            raise SystemExit("No stages enabled in config.")

        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        finally:
            for observer in observers:
                observer.stop()
            for observer in observers:
                observer.join()
            for dispatcher in dispatchers.values():
                dispatcher.shutdown()
            server.close()
            await server.wait_closed()

    except asyncio.CancelledError:
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
