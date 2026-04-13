Watcher files generated for Unreal + Python pipeline.

Files:
- folder_ws_watcher.py
- watcher_config.json

Install:
pip install watchdog websocket-client

Run:
python folder_ws_watcher.py --config watcher_config.json

Optional startup scan:
python folder_ws_watcher.py --config watcher_config.json --startup-scan

WebSocket payload examples:
{
  "type": "input_jpg_ready",
  "full_path": "C:/Pipeline/Input/image00004.jpg",
  "file_name": "image00004.jpg",
  "event": "created",
  "timestamp": 1770000000.0
}

{
  "type": "spawn_json_ready",
  "full_path": "C:/Pipeline/Output/output_output_image00004_vector_spawn_points.json",
  "file_name": "output_output_image00004_vector_spawn_points.json",
  "event": "modified",
  "timestamp": 1770000001.0
}
