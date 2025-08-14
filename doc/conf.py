import os
import sys
import toml
from pathlib import Path

# Add project root and python source to path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../python'))

# Read version from pyproject.toml
pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
pyproject_data = toml.load(str(pyproject_path))
version = pyproject_data["project"]["version"]
release = version

# Project information
project = 'batss'
copyright = '2025, David Doty, Joshua Petrack, and Eric Severson'
author = 'David Doty, Joshua Petrack, and Eric Severson'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # Remove _static to avoid warning

# Autodoc settings
autodoc_member_order = 'bysource'
autoclass_content = 'both'
typehints_fully_qualified = False
always_document_param_types = True

autosummary_generate = True

# Nitpicky mode to catch all references
# nitpicky = True
# nitpick_ignore = [
#     ('py:class', 'ipywidgets.interactive'),
#     ('py:class', 'numpy.uint64'),
#     ('py:class', 'matplotlib.figure.Figure'),
#     ('py:class', 'matplotlib.axes._axes.Axes'),
#     ('py:class', 'gpac.crn.Reaction'),
#     ('py:class', 'gpac.crn.Specie'),
#     ('py:class', 'tqdm.std.tqdm'),
#     ('py:class', 'Returns'),
#     ('py:data', 'Snapshot.simulation.state_list'),
#     ('py:meth', 'Simulation.add_snapshot'),
#     ('py:class', 'batss_rust.SimulatorMultiBatch'),
#     ('py:class', 'batss_rust.SimulatorSequentialArray'),
#     ('py:class', 'SimulatorMultiBatch'),
#     ('py:class', 'SimulatorSequentialArray'),
#     ('py:class', 'Snapshot'),
# ]

# Intersphinx mapping for external references (reduced for speed)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Suppress certain Sphinx warnings
suppress_warnings = [
    # Suppress duplicate object description warnings from autosummary
    'app.add_directive',
]

# Performance optimizations
intersphinx_timeout = 5  # Reduce timeout for external inventories
autodoc_typehints = 'description'  # Faster than signature

# Mock only the heaviest dependencies 
autodoc_mock_imports = [
    'gpac',
    'matplotlib',
    'matplotlib.pyplot',
    'batss_rust',
    'batss.batss_rust',
    'batss.batss_rust.batss_rust',
]

# Disable autosummary to avoid import issues
autosummary_generate = False
