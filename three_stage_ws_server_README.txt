Python-hosted WebSocket server version of the 3-stage watcher.

Install:
pip install websockets watchdog

Run:
python three_stage_pipeline_ws_server.py --config three_stage_ws_server_config.json

Unreal should connect as a WebSocket CLIENT to:
ws://127.0.0.1:8080

Flow:
1) Stage 1 source JPG -> Python broadcasts source_jpg_ready
2) Stage 2 processed JPG -> Python runs run_locked_palette_pipeline.py
3) Stage 3 spawn JSON -> Python broadcasts spawn_json_ready

Notes:
- If no Unreal client is connected, broadcast payloads are logged and dropped.
- Stage 3 keeps the queue/cooldown behavior.
