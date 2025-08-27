# panel serve mini_iso/app.py \
#     --allow-websocket-origin="*:5003" \
#     --port="5003" \
#     --args "mini_iso/datasets/three_zones/case1.json"
poetry run python mini_iso/app.py mini_iso/datasets/three-zones-case1 --port 5003
