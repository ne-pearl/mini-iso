
1. Install Python (`py`) using [python-3.12.3-amd64.exe](https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe)

2. Use `py` to install `pip` and `pipx`:

```bash
# Install pip and pipx (use python or py)
python -m ensurepip --upgrade
python -m pip install --user pipx

# Download Mini-ISO and install dependencies
git clone https://github.com/ne-pearl/mini-iso.git

# Move into mini-iso folder 
cd mini-iso

# Install Mini ISO dependencies with Poetry
python -m pipx install poetry
# On Linux:
poetry install
poetry shell
# On Windows, e.g.:
C:/Users/Jon/pipx/venvs/poetry/Scripts/poetry.exe install
C:/Users/Jon/pipx/venvs/poetry/Scripts/poetry.exe shell

# Create stand-alone executable (--onefile vs --onedir)
# Use --onedir or --onefile
# NB: The two locations of gurobi.lic are for Windows and Mac/Linux
pyinstaller --onefile mini_iso/app.py --name mini-iso --hiddenimport pydantic.deprecated.decorator --hiddenimport pkg_resources.extern --add-data ./gurobi.lic:gurobipy/ --add-data ./gurobi.lic:gurobipy/.libs/ --add-data ./mini_iso/datasets/:./

# Test the new executable
dist/mini_iso.exe three-zones-case1
dist/mini_iso.exe mini-new-england-uniform
dist/mini_iso.exe mini-new-england
```
