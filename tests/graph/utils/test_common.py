# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
from math import sqrt
from torchcp.graph.utils import compute_adj_knn


def test_invalid_compute_adj_knn():
    features = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0]
    ])
    with pytest.raises(ValueError, match="The number of nodes cannot be less than k"):
        compute_adj_knn(features, 6)

def test_compute_adj_knn():
    features = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0]
    ])
    k = 2
    knn_edge, knn_weights = compute_adj_knn(features, k)
    excepted_edge = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                                  [2, 4, 3, 4, 0, 4, 2, 4, 2, 3]])
    excepted_weight = torch.tensor([2/sqrt(6), 2/3, 1/2, 1/sqrt(6), 2/sqrt(6), 2/sqrt(6), 
                                    1/2, 2/sqrt(6), 2/sqrt(6), 2/sqrt(6)])
    assert torch.equal(knn_edge, excepted_edge)
    assert torch.allclose(knn_weights, excepted_weight)