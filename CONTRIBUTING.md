## Developing

To get started, you'll need to create a virtual environment and install the requirements:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package with development dependencies
pip install -e ".[development,testing,examples]"

# Install pre-commit hooks
pre-commit install
```

Or using `uv`:

```bash
# Create a virtual environment
uv venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package with development dependencies
uv pip install -e ".[development,testing,examples]"

# Install pre-commit hooks
pre-commit install
```

The pre-commit hooks will automatically run code formatting and linting tools (ruff, black, isort, pyright) on every commit to ensure consistent style.

If you want to skip pre-commit (for a WIP commit), you can use the `--no-verify` flag:

```bash
git commit --no-verify -m "Your commit message"
```
