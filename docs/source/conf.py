# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../../'))

from unittest.mock import Mock  # noqa: F401, E402
# from sphinx.ext.autodoc.importer import _MockObject as Mock
Mock.Module = object
sys.modules['torch'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['numpy.linalg'] = Mock()
sys.modules['scipy'] = Mock()
sys.modules['scipy.optimize'] = Mock()
sys.modules['scipy.interpolate'] = Mock()
sys.modules['scipy.ndimage'] = Mock()
sys.modules['scipy.ndimage.filters'] = Mock()
sys.modules['tensorflow'] = Mock()
sys.modules['theano'] = Mock()
sys.modules['theano.tensor'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['torch.autograd'] = Mock()
sys.modules['torch.autograd.gradcheck'] = Mock()
sys.modules['torch.distributions'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.nn.functional'] = Mock()
sys.modules['torch.optim'] = Mock()
sys.modules['torch.nn.modules'] = Mock()
sys.modules['torch.nn.modules.utils'] = Mock()
sys.modules['torch.nn.modules.loss'] = Mock()
sys.modules['torch.utils'] = Mock()
sys.modules['torch.utils.model_zoo'] = Mock()
sys.modules['torch.nn.init'] = Mock()
sys.modules['torch.utils.data'] = Mock()
sys.modules['torchvision'] = Mock()
sys.modules['randomstate'] = Mock()
sys.modules['scipy._lib'] = Mock()
sys.modules['sklearn.cluster'] = Mock()
import torchcp

project = 'TorchCP'
copyright = '2023, ml-stat-Sustech'
author = 'ml-stat-Sustech'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]




