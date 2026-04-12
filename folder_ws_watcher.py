#!/usr/bin/env python3
"""
folder_ws_watcher.py

Watches:
1) an input image folder for a specific JPG filename
2) a spawn-output folder for vector spawn JSON files matching a filename pattern

When a matching file is found/changed and appears stable, sends a WebSocket JSON
message to Unreal containing:
- type
- full_path
- file_name
- event
- timestamp

Default event types:
- "input_jpg_ready"
- "spawn_json_ready"

Intended for packaged Unreal exe <-> Python pipeline workflows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
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


# ============================================================
# Config
# ============================================================

DEFAULT_CONFIG = {
    "websocket": {
        "url": "ws://127.0.0.1:8080",
        "connect_on_start": True,
        "reconnect_on_send_failure": True
    },
    "watcher": {
        "recursive": False,
        "debounce_seconds": 0.5,
        "poll_interval_seconds": 1.0,
        "stable_check_interval_seconds": 0.2,
        "stable_checks_required": 3,
        "log_level": "INFO"
    },
    "jpg_watch": {
        "enabled": True,
        "folder": r"C:\\Pipeline\\Input",
        "target_file_name": "image00004.jpg",
        "case_sensitive": False,
        "event_type": "input_jpg_ready"
    },
    "spawn_json_watch": {
        "enabled": True,
        "folder": r"C:\\Pipeline\\Output",
        "file_regex": r"^output_output_.*_vector_spawn_points\.json$",
        "case_sensitive": False,
        "event_type": "spawn_json_ready"
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


# ============================================================
# WebSocket sender
# ============================================================

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


# ============================================================
# File matching / stability
# ============================================================

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


def wait_until_file_stable(
    path: Path,
    check_interval_seconds: float,
    stable_checks_required: int,
) -> bool:
    """
    Returns True once file size and mtime stop changing across N checks.
    Returns False if file disappears before stabilizing.
    """
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


# ============================================================
# Watch target
# ============================================================

@dataclass
class WatchTarget:
    name: str
    folder: Path
    matcher: Callable[[Path], bool]
    event_type: str
    debounce_seconds: float
    stable_check_interval_seconds: float
    stable_checks_required: int


class RoutedEventHandler(FileSystemEventHandler):
    def __init__(self, target: WatchTarget, send_func: Callable[[dict], None]):
        super().__init__()
        self.target = target
        self.send_func = send_func
        self.last_sent_time_by_path: dict[str, float] = {}
        self.lock = threading.Lock()

    def _debounced(self, path_str: str) -> bool:
        now = time.time()
        with self.lock:
            last = self.last_sent_time_by_path.get(path_str, 0.0)
            if now - last < self.target.debounce_seconds:
                return True
            self.last_sent_time_by_path[path_str] = now
        return False

    def _maybe_send(self, file_path: Path, event_name: str) -> None:
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
            "timestamp": time.time()
        }
        self.send_func(payload)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_send(Path(event.src_path), "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_send(Path(event.src_path), "modified")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._maybe_send(Path(event.dest_path), "moved")


# ============================================================
# Boot
# ============================================================

def build_targets(cfg: dict) -> list[WatchTarget]:
    watcher_cfg = cfg["watcher"]
    targets: list[WatchTarget] = []

    jpg_cfg = cfg.get("jpg_watch", {})
    if jpg_cfg.get("enabled", True):
        folder = Path(jpg_cfg["folder"]).expanduser()
        matcher = make_exact_name_matcher(
            target_file_name=str(jpg_cfg["target_file_name"]),
            case_sensitive=bool(jpg_cfg.get("case_sensitive", False)),
        )
        targets.append(
            WatchTarget(
                name="jpg_watch",
                folder=folder,
                matcher=matcher,
                event_type=str(jpg_cfg.get("event_type", "input_jpg_ready")),
                debounce_seconds=float(watcher_cfg.get("debounce_seconds", 0.5)),
                stable_check_interval_seconds=float(watcher_cfg.get("stable_check_interval_seconds", 0.2)),
                stable_checks_required=int(watcher_cfg.get("stable_checks_required", 3)),
            )
        )

    spawn_cfg = cfg.get("spawn_json_watch", {})
    if spawn_cfg.get("enabled", True):
        folder = Path(spawn_cfg["folder"]).expanduser()
        matcher = make_regex_name_matcher(
            pattern=str(spawn_cfg["file_regex"]),
            case_sensitive=bool(spawn_cfg.get("case_sensitive", False)),
        )
        targets.append(
            WatchTarget(
                name="spawn_json_watch",
                folder=folder,
                matcher=matcher,
                event_type=str(spawn_cfg.get("event_type", "spawn_json_ready")),
                debounce_seconds=float(watcher_cfg.get("debounce_seconds", 0.5)),
                stable_check_interval_seconds=float(watcher_cfg.get("stable_check_interval_seconds", 0.2)),
                stable_checks_required=int(watcher_cfg.get("stable_checks_required", 3)),
            )
        )

    return targets


def optional_startup_scan(targets: list[WatchTarget], send_func: Callable[[dict], None]) -> None:
    """
    Optional helper so if the file already exists before the watcher starts,
    you can still emit one startup message.
    """
    for target in targets:
        if not target.folder.exists():
            continue
        for entry in target.folder.iterdir():
            if entry.is_file() and target.matcher(entry):
                payload = {
                    "type": target.event_type,
                    "full_path": str(entry.resolve()),
                    "file_name": entry.name,
                    "event": "startup_scan",
                    "timestamp": time.time()
                }
                send_func(payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch folders and send matching file events to Unreal over WebSocket.")
    ap.add_argument(
        "--config",
        default="watcher_config.json",
        help="Path to watcher config JSON."
    )
    ap.add_argument(
        "--startup-scan",
        action="store_true",
        help="Also send messages for already-existing matching files when starting."
    )
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

    watcher_cfg = cfg["watcher"]
    recursive = bool(watcher_cfg.get("recursive", False))
    poll_interval = float(watcher_cfg.get("poll_interval_seconds", 1.0))

    targets = build_targets(cfg)
    if not targets:
        raise SystemExit("No watch targets enabled in config.")

    observers: list[Observer] = []

    try:
        for target in targets:
            target.folder.mkdir(parents=True, exist_ok=True)

            handler = RoutedEventHandler(target=target, send_func=sender.send_json)
            observer = Observer(timeout=poll_interval)
            observer.schedule(handler, str(target.folder), recursive=recursive)
            observer.start()
            observers.append(observer)

            logging.info("Watching [%s] folder: %s", target.name, str(target.folder.resolve()))

        if args.startup_scan:
            optional_startup_scan(targets, sender.send_json)

        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("Stopping watcher...")
    finally:
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()
        sender.close()


if __name__ == "__main__":
    main()
