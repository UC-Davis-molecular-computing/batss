import os
import sys
import toml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

# Read version from pyproject.toml
pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
pyproject_data = toml.load(str(pyproject_path))
version = pyproject_data["project"]["version"]
release = version

# Project information
project = 'ppsim'
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
html_static_path = ['_static']

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
#     ('py:class', 'ppsim_rust.SimulatorMultiBatch'),
#     ('py:class', 'ppsim_rust.SimulatorSequentialArray'),
#     ('py:class', 'SimulatorMultiBatch'),
#     ('py:class', 'SimulatorSequentialArray'),
#     ('py:class', 'Snapshot'),
# ]

# Intersphinx mapping for external references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'ipywidgets': ('https://ipywidgets.readthedocs.io/en/latest/', None),
}

# Suppress certain Sphinx warnings
suppress_warnings = [
    # Suppress duplicate object description warnings from autosummary
    'app.add_directive',
]
