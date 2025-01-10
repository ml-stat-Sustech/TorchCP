# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

with open(os.path.join(os.path.dirname(__file__), 'VERSION'), encoding="utf-8") as f:
    __version__ = f.read().strip()

from . import classification
from . import regression
from . import graph
from . import llm
