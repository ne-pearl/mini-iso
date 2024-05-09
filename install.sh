#!/usr/bin/bash

# Use --onedir or --onefile
pyinstaller --onefile \
    mini_iso/app.py \
    --name mini_iso \
    --hiddenimport pydantic.deprecated.decorator \
    --add-data ./gurobi.lic:gurobipy/.libs/ \
    --add-data ./mini_iso/datasets/:datasets/

echo "To run, use: ./dist/app/app path/to/data.json"
echo "Examples:"
echo "$ dist/mini_iso datasets/three_zones/case1.json"
echo "$ dist/mini_iso datasets/mini_new_england/mini_new_england.json"
