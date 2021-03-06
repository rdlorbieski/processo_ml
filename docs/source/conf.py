# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


sys.path.insert(0, os.path.abspath('.'))

sys.path.insert(0, os.path.abspath('../'))

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'FlaskApp'
copyright = '2021, Rodolfo Lorbieski'
author = 'Rodolfo Lorbieski'
version = '1.2.1'
# The full version, including alpha/beta/rc tags
release = '1.2.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# Para trocar de tema olhe a documentação em: https://www.sphinx-doc.org/en/master/usage/theming.html
# Consulte mais sobre temas em: https://sphinx-themes.org/
# options = ['scrolls', 'nature', 'bizstyle', 'classic', 'haiku', 'sphinxdoc', 'alabaster', 'pyramid', 'epub', 'agogo', 'traditional', 'basic']
html_theme = 'scrolls'


# Configuração baseada no tema em questão: no exemplo abaixo é usado para o tema 'classic'
#html_theme_options = {
#    "rightsidebar": "true",
#    "relbarbgcolor": "black"
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

