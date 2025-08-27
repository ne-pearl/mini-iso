# panel serve mini_iso/app.py \
#     --allow-websocket-origin="*:5001" \
#     --port="5001" \
#     --args "mini_iso/datasets/one_zone/one_zone.json"
poetry run python mini_iso/app.py mini_iso/datasets/one-zone --port 5001
