# Copyright (c) 2023-present, SUSTech-ML.
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
      url='https://github.com/ml-stat-Sustech/TorchCP',
      package_data={'examples': ['*.ipynb']},
      description="A Python toolbox for conformal prediction on deep learning models.",
      install_requires=[
          "torch>=2.1.0",
          "torchvision>=0.16.0",
          "transformers>=4.20.0",
          "numpy>=1.20.0",
          "setuptools>=59.5.0",
          "tqdm>=4.60.0",
          "scikit-learn>=1.5.0",
          "pandas>=2.1.4",
          "Pillow>=10.3.0",
          "torch_geometric>=2.4.0",
          "torchsort>=0.1.9"],
      include_package_data=True,
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )
