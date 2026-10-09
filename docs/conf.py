# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata

project = 'mcpy'
author = 'Riccardo Farris, Emanuele Telari, Konstantin M. Neyman, Albert Bruix'
copyright = f'2023-2026, {author}'
# Every docs builder installs mcpy (autodoc imports it), so the installed
# metadata is the single source of the version.
release = importlib.metadata.version('mcpy-gcmc')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon',
              'sphinx.ext.todo', 'sphinx.ext.viewcode']

# The optional backends are not installed on the docs builder, and importing a
# real torch here is both slow and, in a misconfigured env, fatal (SIGBUS).
# Mocking lets autodoc read the wrappers' signatures and docstrings anyway.
autodoc_mock_imports = ['torch', 'mace', 'nvalchemi', 'cuequivariance', 'mpi4py']
autodoc_member_order = 'bysource'
# Parameters are documented in __init__, so render both docstrings.
autoclass_content = 'both'
autodoc_typehints = 'description'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Every cross-reference must resolve, so -W catches a broken one. Types from
# ASE, NumPy, torch and the standard library are not documented here and are
# ignored by name.
nitpicky = True
nitpick_ignore_regex = [
    (r'py:.*', r'(ase|numpy|np|torch|mace|nvalchemi|mpi4py|logging)(\..*)?'),
    (r'py:class', r'(optional|callable|MACEWrapper)'),
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
