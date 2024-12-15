# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
from math import sqrt
import torch
from torch_geometric.data import Data

from torchcp.classification.score import THR
from torchcp.graph.score import SNAPS


@pytest.fixture
def graph_data():
    x = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ], dtype=torch.float32)

    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)

    y = torch.tensor([0, 1, 0], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_weight=None, y=y)

    return data


@pytest.fixture
def base_score_function():
    return THR(score_type="softmax")


@pytest.mark.parametrize("xi", [-0.1, 1.1])
@pytest.mark.parametrize("mu", [-0.1, 1.1])
def test_invalid_lambda_muues(graph_data, base_score_function, xi, mu):
    with pytest.raises(ValueError, match="The parameter 'xi' must be a value between 0 and 1."):
        SNAPS(graph_data, base_score_function, xi=xi)

    with pytest.raises(ValueError, match="The parameter 'mu' must be a value between 0 and 1."):
        SNAPS(graph_data, base_score_function, mu=mu)

    with pytest.raises(ValueError, match="The summation of 'xi' and 'mu' must not be greater than 1."):
        SNAPS(graph_data, base_score_function, xi=0.6, mu=0.6)


def test_valid_initialization(graph_data, base_score_function):
    model = SNAPS(graph_data, base_score_function, xi=0.3, mu=0.3)
    assert model._xi == 0.3
    assert model._mu == 0.3

    model = SNAPS(graph_data, base_score_function, features=graph_data.x, k=2)
    excepted_adjknn = torch.tensor([[0, 11/(5 * sqrt(5)), 17 / sqrt(305)],
                                    [11/(5 * sqrt(5)), 0, 39 / (5 * sqrt(61))],
                                    [17 / sqrt(305), 39 / (5 * sqrt(61)), 0]])
    assert torch.allclose(model._adj_knn.to_dense(), excepted_adjknn)


def test_knn_processing(graph_data, base_score_function):
    def are_sparse_tensors_equal(tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            return False
        if tensor1.device != tensor2.device:
            return False
        if not torch.equal(tensor1.coalesce().indices(), tensor2.coalesce().indices()):
            return False
        if not torch.equal(tensor1.coalesce().values(), tensor2.coalesce().values()):
            return False
        return True

    knn_edge = torch.tensor([
        [0, 1, 2],
        [1, 2, 0]
    ], dtype=torch.long)
    knn_weight = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    adj_knn = torch.sparse_coo_tensor(
        knn_edge,
        knn_weight,
        (graph_data.num_nodes, graph_data.num_nodes))
    knn_degs = torch.matmul(adj_knn, torch.ones((adj_knn.shape[0])))

    score_function = SNAPS(graph_data, base_score_function, knn_edge=knn_edge, knn_weight=knn_weight)
    assert are_sparse_tensors_equal(score_function._adj_knn, adj_knn)
    assert torch.equal(score_function._knn_degs, knn_degs)

    score_function = SNAPS(graph_data, base_score_function, knn_edge=None, knn_weight=knn_weight)
    assert score_function._adj_knn is None

    score_function = SNAPS(graph_data, base_score_function, knn_edge=knn_edge, knn_weight=None)

    knn_weight = torch.ones(knn_edge.shape[1])
    adj_knn = torch.sparse_coo_tensor(
        knn_edge,
        knn_weight,
        (graph_data.num_nodes, graph_data.num_nodes))
    knn_degs = torch.matmul(adj_knn, torch.ones((adj_knn.shape[0])))
    assert are_sparse_tensors_equal(score_function._adj_knn, adj_knn)
    assert torch.equal(score_function._knn_degs, knn_degs)


def test_snaps_call_without_labels(graph_data, base_score_function):
    knn_edge = torch.tensor([
        [0, 1, 2],
        [1, 2, 0]
    ], dtype=torch.long)
    knn_weight = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    snaps = SNAPS(graph_data, base_score_function, xi=0.4, mu=0.25, knn_edge=knn_edge,
                  knn_weight=knn_weight)

    logits = torch.tensor([
        [1.0, 0.5],
        [0.2, 0.8],
        [0.4, 0.6]
    ], dtype=torch.float32)

    base_scores = base_score_function(logits)
    similarity_scores = torch.tensor([
        [base_scores[1, 0], base_scores[1, 1]],
        [base_scores[2, 0], base_scores[2, 1]],
        [base_scores[0, 0], base_scores[0, 1]]
    ], dtype=torch.float32)
    neigh_scores = torch.tensor([
        [base_scores[1, 0], base_scores[1, 1]],
        [(base_scores[0, 0] + base_scores[2, 0]) / 2, (base_scores[0, 1] + base_scores[2, 1]) / 2],
        [base_scores[1, 0], base_scores[1, 1]]
    ], dtype=torch.float32)

    expected_scores = 0.35 * base_scores + 0.4 * similarity_scores + 0.25 * neigh_scores

    scores = snaps(logits)
    assert torch.allclose(scores, expected_scores, atol=1e-5)


def test_snaps_call_with_labels(graph_data, base_score_function):
    knn_edge = torch.tensor([
        [0, 1, 2],
        [1, 2, 0]
    ], dtype=torch.long)
    knn_weight = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    snaps = SNAPS(graph_data, base_score_function, xi=0.4, mu=0.25, knn_edge=knn_edge,
                  knn_weight=knn_weight)

    logits = torch.tensor([
        [1.0, 0.5],
        [0.2, 0.8],
        [0.4, 0.6]
    ], dtype=torch.float32)

    labels = torch.tensor([0, 1, 0], dtype=torch.long)

    scores = snaps(logits, labels)
    expected_scores = snaps(logits)
    assert torch.allclose(scores, expected_scores[torch.arange(3), labels], atol=1e-5)
