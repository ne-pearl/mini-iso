#!/usr/bin/bash

# Use --onedir or --onefile
pyinstaller --onefile \
    mini_iso/app.py \
    --hiddenimport pydantic.deprecated.decorator \
    --add-data ./gurobi.lic:gurobipy/.libs/

# Additional text file not picked up by pyinstaller
cp gurobi.lic dist/app/_internal/gurobipy/.libs/.

echo "To run, use: ./dist/app/app absolute/path/to/data.json"
echo "Example:"
echo "./dist/app/app \\"
echo "\$(realpath mini_iso/datasets/mini_new_england/mini_new_england.json)"