
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
# On Windows: 
~/pipx/venvs/poetry/Scripts/poetry.exe install
~/pipx/venvs/poetry/Scripts/poetry.exe shell

# Create stand-alone executable (--onefile vs --onedir)
pyinstaller --onefile mini_iso/app.py --name mini_iso --hiddenimport pydantic.deprecated.decorator --add-data ./gurobi.lic:gurobipy/ --add-data ./gurobi.lic:gurobipy/.libs/ --add-data ./mini_iso/datasets/:datasets/

# Test the new executable
dist/mini_iso datasets/three_zones/case1.json
dist/mini_iso datasets/mini_new_england/mini_new_england.json
```
