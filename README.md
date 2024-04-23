# Mini-ISO 

Simulates electricity market auctions for classroom use.

## Installation

### Step 1. Install `git` and `pipx`

#### MacOS

```bash
git --version  # triggers prompts to install git
brew install pipx
pipx ensurepath
sudo pipx ensurepath --global
```

#### Windows

1. Install Python (`py`) using one of the following installers:
   * [python-3.11.9-amd64.exe](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
   * [python-3.12.3-amd64.exe](https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe)

2. Use `py` to install `pip` and `pipx`:

```bash
py -m ensurepip --upgrade
py -m pip install --user pipx
```

#### Linux (Ubuntu)

```bash
sudo apt update
sudo apt install git-all
sudo apt install pipx
pipx ensurepath
sudo pipx ensurepath --global
```

### Step2: Install Mini ISO

```bash
# Download Mini-ISO and install dependencies
git clone https://github.com/ne-pearl/mini-iso.git

# Move into mini-iso folder 
cd mini-iso

# Install Mini ISO dependencies with Poetry
pipx install poetry
poetry env use 3.11  # or 3.12 etc.
poetry install
```

---

## Running Mini ISO

```bash
cd your/path/to/mini-iso
poetry shell
panel serve mini_iso/app.py --port 5001 \
    --args mini_iso/datasets/one_zone/one_zone.json
```
