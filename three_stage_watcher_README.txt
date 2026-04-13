Three-stage watcher created.

Stage 1:
- watches source JPG folder
- sends WebSocket payload to Unreal

Stage 2:
- watches processed JPG folder
- runs run_locked_palette_pipeline.py on matching file
- uses locked_palette_pipeline_config.json

Stage 3:
- watches output folder for spawn JSON
- sends WebSocket payload to Unreal
- queue/cooldown supported here

Install:
pip install watchdog websocket-client

Run:
python three_stage_pipeline_watcher.py --config three_stage_watcher_config.json
