#!/bin/bash

# Use --onedir or --onefile
# NB: The two locations of gurobi.lic are for Windows and Mac/Linux
# NB: `pkg_resources.extern` is missing on Windows (May 2024)
pyinstaller --onefile mini_iso/app.py --name mini-iso --hiddenimport pydantic.deprecated.decorator --hiddenimport pkg_resources.extern --add-data ./gurobi.lic:gurobipy/ --add-data ./gurobi.lic:gurobipy/.libs/ --add-data ./mini_iso/datasets/:./

echo ""
echo "To run, use: ./dist/app/app absolute/path/to/data.json"
echo "Examples:"
echo "$ dist/mini-iso three-nodes"
echo "$ dist/mini-iso mini-new-england-uni"
echo "$ dist/mini-iso mini-new-england-multi"
echo ""
