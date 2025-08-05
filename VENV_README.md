# Virtual Environment Setup for batss

This document provides instructions for setting up and using a virtual environment for the batss package.

## Setup Instructions

### 1. Create a Virtual Environment

```powershell
# Create a virtual environment named .venv in the project directory
python -m venv .venv
```

### 2. Activate the Virtual Environment

```powershell
# In PowerShell
.\.venv\Scripts\Activate.ps1

# In Command Prompt
.\.venv\Scripts\activate.bat
```

### 3. Install Required Packages

```powershell
# Install maturin
pip install maturin

# If you're using Conda, temporarily unset the CONDA_PREFIX environment variable
$env:CONDA_PREFIX = $null
```

### 4. Build and Install the Package

```powershell
# Build the Rust extension
maturin develop

# Install the Python package in development mode
pip install -e .
```

### 5. For Jupyter Notebook Support

```powershell
# Install ipykernel
pip install ipykernel

# Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=batss-venv --display-name="Python (batss-venv)"
```

## Usage

### Running Python Scripts

To run Python scripts with the virtual environment:

```powershell
# Activate the virtual environment first
.\.venv\Scripts\Activate.ps1

# Then run your script
python your_script.py
```

Or directly:

```powershell
.\.venv\Scripts\python.exe your_script.py
```

### Using Jupyter Notebooks

When opening a Jupyter notebook, select the "Python (batss-venv)" kernel from the kernel dropdown menu.

## Troubleshooting

### Conda and Virtual Environment Conflicts

If you're using Conda and encounter an error like:

```
Both VIRTUAL_ENV and CONDA_PREFIX are set. Please unset one of them
```

Use the following command before running maturin:

```powershell
$env:CONDA_PREFIX = $null
```

### Import Errors

If you encounter import errors when trying to use the batss package, make sure:

1. The virtual environment is activated
2. The package is installed in development mode (`pip install -e .`)
3. The Rust extension is built (`maturin develop`)

## Notes

- The virtual environment isolates the batss package and its dependencies from your global Python installation
- The development mode installation (`pip install -e .`) allows you to make changes to the Python code without reinstalling
- When you make changes to the Rust code, you need to rebuild the extension with `maturin develop`
