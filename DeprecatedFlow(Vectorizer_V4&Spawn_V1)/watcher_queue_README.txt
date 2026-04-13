Updated watcher with per-target delayed queues.

Behavior:
- First matching file during an idle period is sent immediately.
- Then that target enters cooldown.
- During cooldown, later matching files are queued.
- When cooldown ends, queued files are flushed in FIFO order.
- Each watch target has its own independent queue.

Current config:
- jpg_watch queue disabled
- spawn_json_watch queue enabled with 30 second cooldown

Install:
pip install watchdog websocket-client

Run:
python folder_ws_watcher_with_queue.py --config watcher_config_with_queue.json

Payload now also includes:
- queue_name
