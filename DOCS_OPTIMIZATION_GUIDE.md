# Read the Docs Optimization Guide

This guide documents how to optimize Read the Docs builds for Python packages with Rust extensions and heavy dependencies, reducing build times from 2+ minutes to ~20 seconds while generating complete API documentation.

## Problem

Standard Read the Docs builds for packages with:
- Rust extensions requiring compilation
- Heavy dependencies (scipy, matplotlib, pandas, etc.)
- Modern Python type annotations (`|` union syntax)

...can take 2+ minutes and often fail to generate proper API documentation due to import errors.

## Solution Overview

1. **Use uv** for faster dependency resolution and installation
2. **Mock heavy dependencies** that aren't needed for documentation
3. **Include essential dependencies** needed for type annotations to work
4. **Skip package installation** to avoid Rust compilation
5. **Let Sphinx read source directly** with selective mocking

## Implementation

### 1. `.readthedocs.yaml` Configuration

```yaml
version: 2

sphinx:
   configuration: doc/conf.py
   builder: html
   fail_on_warning: false

build:
   os: ubuntu-latest
   tools:
      python: "3.13"
   jobs:
      pre_create_environment:
         - asdf plugin add uv
         - asdf install uv latest
         - asdf global uv latest
      create_environment:
         - uv venv "${READTHEDOCS_VIRTUALENV_PATH}"
      install:
         - uv pip install -r docs-requirements.txt
```

**Key points:**
- Uses Ubuntu 24.04 for better performance
- Installs uv for faster package management
- **Does NOT install the package** (skips Rust compilation)
- Uses a minimal requirements file

### 2. `docs-requirements.txt`

```txt
# Docs requirements - include deps needed for union syntax
sphinx>=7.0.0
sphinx-rtd-theme
sphinx-autodoc-typehints
toml
# Essential for union type annotations to work
numpy>=2
pandas>=2
natsort>=8
tqdm>=4
# Skip only the heaviest: gpac (brings scipy, matplotlib)
```

**Strategy:**
- Include **only** dependencies needed for type annotations (`pd.MultiIndex | list[str]`)
- Exclude the heaviest packages that cause long build times
- Keep the bare minimum for Sphinx + essential runtime types

### 3. `doc/conf.py` Sphinx Configuration

```python
# Mock only the heaviest dependencies 
autodoc_mock_imports = [
    'gpac',
    'matplotlib',
    'matplotlib.pyplot', 
    'batss_rust',           # Your Rust extension
    'your_package.rust_ext', # Adjust to your package
]

# Performance optimizations
intersphinx_timeout = 5  # Reduce timeout for external inventories
autodoc_typehints = 'description'  # Faster than signature

# Disable autosummary to avoid import issues (optional)
autosummary_generate = False
```

**Key decisions:**
- **Mock Rust extensions** (avoids compilation requirement)
- **Mock heavy packages** like matplotlib, scipy if not needed for types
- **Don't mock** numpy, pandas if you use them in union types
- Set fast timeouts and efficient settings

### 4. Handle Version Import Issues

If your `__init__.py` tries to get version from package metadata:

```python
# Before (fails when package not installed)
from importlib.metadata import version
__version__ = version("your_package")

# After (with fallback)
try:
    from importlib.metadata import version
    __version__ = version("your_package")
except Exception:
    # Fallback for docs builds where package isn't installed
    __version__ = "1.0.0"  # Use actual version
```

## Results

**Before:**
- Build time: 2+ minutes
- Dependencies: 140+ packages
- Rust compilation: ~60s
- Documentation: Often empty due to import failures

**After:**
- Build time: ~20 seconds
- Dependencies: ~35 packages  
- Rust compilation: 0s (skipped)
- Documentation: Complete with full API docs

## Key Insights

1. **Modern type annotations require real types** - you can't mock `pandas` if you use `pd.MultiIndex | list[str]`

2. **Rust compilation is the biggest bottleneck** - avoiding package installation saves ~60s

3. **Selective mocking works better than aggressive mocking** - mock only what's actually slow/unnecessary

4. **uv provides significant speedup** - especially for dependency resolution

5. **Source-based documentation** - Sphinx can generate docs from source files without installing the package

## Troubleshooting

**Empty API documentation:**
- Check that essential type dependencies are installed, not mocked
- Verify your package's `__init__.py` doesn't fail on import
- Look for union type syntax that requires actual types

**Build failures:**
- Start with minimal mocking and add more gradually
- Check Read the Docs build logs for specific import errors
- Test locally first with the same requirements

**Slow builds despite optimization:**
- Profile which dependencies take longest to install
- Consider if any dependencies can be mocked instead of installed
- Check if your package has circular dependencies

## Adaptation for Other Projects

1. **Identify your heaviest dependencies** - use `pip list` or `uv pip list` to see what's installed
2. **Find your type annotation dependencies** - look for `|` unions in your code
3. **Mock your compiled extensions** - Rust, C, Fortran, etc.
4. **Test locally first** - use the same venv setup to verify docs generate properly
5. **Iterate on the balance** - between speed and completeness

This approach should work for any Python package with similar challenges: compiled extensions, heavy dependencies, and modern type annotations.