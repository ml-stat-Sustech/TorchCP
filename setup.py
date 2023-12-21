# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from setuptools import find_packages
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'torchcp/VERSION')) as f:
    version = f.read().strip()

setup(name='torchcp',
      version=version,
      url='https://github.com/ml-stat-Sustech/DeepCP',
      package_data={'torchcp_examples': ['*.ipynb']},
      description="A Python toolbox for conformal prediction research on deep learning models.",
      install_requires=[],
      include_package_data=True,
      packages=find_packages())
