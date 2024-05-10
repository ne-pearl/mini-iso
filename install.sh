#!/usr/bin/bash

# Use --onedir or --onefile
pyinstaller --onefile \
    mini_iso/app.py \
    --name mini_iso \
    --hiddenimport pydantic.deprecated.decorator \
    --add-data ./gurobi.lic:gurobipy/.libs/ \
    --add-data ./mini_iso/datasets/:datasets/

echo ""
echo "To run, use: ./dist/app/app absolute/path/to/data.json"
echo "Examples:"
echo "$ dist/mini_iso datasets/one_zone/one_zone.json"
echo "$ dist/mini_iso datasets/one_zones/case1.json"
echo "$ dist/mini_iso datasets/one_zones/case2.json"
echo "$ dist/mini_iso datasets/one_zones/case3.json"
echo "$ dist/mini_iso datasets/mini_new_england/mini_new_england.json"
echo ""
