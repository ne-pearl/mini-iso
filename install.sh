#!bash

# Use --onedir or --onefile
# NB: The two locations of gurobi.lic are for Windows and Mac/Linux
pyinstaller --onefile \
    mini_iso/app.py \
    --name mini_iso \
    --hiddenimport pydantic.deprecated.decorator \
    --add-data ./gurobi.lic:gurobipy/ \
    --add-data ./gurobi.lic:gurobipy/.libs/ \
    --add-data ./mini_iso/datasets/:./

echo ""
echo "To run, use: ./dist/app/app absolute/path/to/data.json"
echo "Examples:"
echo "$ dist/mini_iso three-zones-case1  # 1-4"
echo "$ dist/mini_iso mini-new-england-uniform"
echo "$ dist/mini_iso mini-new-england"
echo ""
