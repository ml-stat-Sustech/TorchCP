# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../../'))

from unittest.mock import Mock  # noqa: F401, E402

# from sphinx.ext.autodoc.importer import _MockObject as Mock
# Mock.Module = object
# sys.modules['torch'] = Mock()
# sys.modules['numpy'] = Mock()
# sys.modules['numpy.linalg'] = Mock()
# sys.modules['scipy'] = Mock()
# sys.modules['scipy.optimize'] = Mock()
# sys.modules['scipy.interpolate'] = Mock()
# sys.modules['scipy.ndimage'] = Mock()
# sys.modules['scipy.ndimage.filters'] = Mock()
# sys.modules['tensorflow'] = Mock()
# sys.modules['theano'] = Mock()
# sys.modules['theano.tensor'] = Mock()
# sys.modules['torch'] = Mock()
# sys.modules['torch.autograd'] = Mock()
# sys.modules['torch.autograd.gradcheck'] = Mock()
# sys.modules['torch.distributions'] = Mock()
# sys.modules['torch.nn'] = Mock()
# sys.modules['torch.nn.functional'] = Mock()
# sys.modules['torch.optim'] = Mock()
# sys.modules['torch.nn.modules'] = Mock()
# sys.modules['torch.nn.modules.utils'] = Mock()
# sys.modules['torch.nn.modules.loss'] = Mock()
# sys.modules['torch.utils'] = Mock()
# sys.modules['torch.utils.model_zoo'] = Mock()
# sys.modules['torch.nn.init'] = Mock()
# sys.modules['torch.utils.data'] = Mock()
# sys.modules['torchvision'] = Mock()
# sys.modules['randomstate'] = Mock()
# sys.modules['scipy._lib'] = Mock()
# sys.modules['sklearn.cluster'] = Mock()
import torchcp

project = 'TorchCP'
copyright = f'{date.today().year}, ml-stat-Sustech'
author = 'ml-stat-Sustech'
with open(os.path.join(os.path.abspath('../../'), 'torchcp/VERSION')) as f:
    version = f.read().strip()

# The full version, including alpha/beta/rc tags
# from openbox import version as _version
# release = str(_version)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# https://www.sphinx-doc.org/en/master/usage/extensions/index.html
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'myst_parser',  # https://myst-parser.readthedocs.io
    'sphinx_copybutton',  # https://sphinx-copybutton.readthedocs.io
    'notfound.extension',  # https://sphinx-notfound-page.readthedocs.io
    'sphinx.ext.autosectionlabel'
]

# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
# example:
#     rst:   :ref:`design principle <overview/overview:design principle>`
#     md:    {ref}`design principle <overview/overview:design principle>`
#            or [](<overview/overview:design principle>)  (hoverxref CANNOT identify this syntax!)
# Make sure the target is unique
autosectionlabel_prefix_document = True  # ref example: `dir/file:header`
autosectionlabel_maxdepth = None  # Must be None. Or failed to build change_logs

# myst_parser
# documentation: https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",  # pip install linkify-it-py
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-header-anchors
myst_heading_anchors = 3  # e.g., [](../overview/overview.md#design-principle) (hoverxref CANNOT identify this syntax!)

# Show tooltip when hover on the reference. Currently, only Read the Docs is supported as backend server!
# https://sphinx-hoverxref.readthedocs.io/
extensions += ['hoverxref.extension']
hoverxref_auto_ref = True
hoverxref_role_types = {
    'ref': 'tooltip',
}
hoverxref_default_type = 'tooltip'  # 'modal' or 'tooltip'
# hoverxref_sphinxtabs = True
hoverxref_mathjax = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

language = 'en'
root_doc = 'index'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_logo = '../../logo.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# default theme
# html_theme = 'alabaster'

# sphinx_rtd_theme (pip install sphinx_rtd_theme) (https://sphinx-rtd-theme.readthedocs.io/)
# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme_options = {
#     'logo_only': True,
#     'style_nav_header_background': 'black',
# }

# furo (pip install furo) (https://pradyunsg.me/furo/)

if os.environ.get('READTHEDOCS') != 'True':
    try:
        import sphinx_rtd_theme
    except ImportError:
        pass  # assume we have sphinx >= 1.3
    html_theme = 'sphinx_rtd_theme'
