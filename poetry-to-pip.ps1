#
# This script is used to generate a regular pip requirements.txt file from the poetry pyproject.toml
#

# 0. Choose a Python version allowed by the project (>=3.10,<3.13)
py --version                     # Windows (or: python3 --version on macOS/Linux)
py -3.12 -m venv .venv           # e.g., use Python 3.11
# macOS/Linux: python3.11 -m venv .venv

# 1. Activate the venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # macOS/Linux

# 2. Upgrade base tools
python -m pip install -U pip setuptools wheel

# 3. Export Poetry deps to requirements.txt (no hashes; includes extras like dask[dataframe])
# Either...
pipx upgrade poetry
# or...
# pip install --user -U poetry 
# or ...
# brew upgrade poetry
poetry --version # We need version ≥1.2.0
poetry self add poetry-plugin-export
poetry export -f requirements.txt --without-hashes -o requirements.txt

# Append our project to the requirements.txt
# NB: In PowerShell, `>>` implies "UTF-16", which is not what we want
echo '-e .' | Out-File -Append -Encoding utf8 requirements.txt

# 5. Install with pip
pip install -r requirements.txt

