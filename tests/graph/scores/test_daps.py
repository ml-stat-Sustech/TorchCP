import pytest

import torch
from torch_geometric.data import Data

from torchcp.classification.scores import THR
from torchcp.graph.scores import DAPS


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


def test_daps_initialization(graph_data, base_score_function):
    daps = DAPS(graph_data, base_score_function, neigh_coef=0.7)
    assert daps._n_vertices == graph_data.num_nodes
    assert daps._device == graph_data.edge_index.device

    if graph_data.edge_weight is None:
        edge_weight = torch.ones(graph_data.edge_index.shape[1])
    else:
        edge_weight = graph_data.edge_weight

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
    
    adj = torch.sparse_coo_tensor(
            graph_data.edge_index,
            edge_weight,
            (graph_data.num_nodes, graph_data.num_nodes))
    assert are_sparse_tensors_equal(daps._adj, adj)

    degs = torch.matmul(adj, torch.ones((adj.shape[0])))
    assert daps._degs.equal(degs)


@pytest.mark.parametrize("neigh_coef", [-0.1, 1.1, -5, 1.5])
def test_invalid_neigh_coef(graph_data, base_score_function, neigh_coef):
    with pytest.raises(ValueError, match="The parameter 'neigh_coef' must be a value between 0 and 1."):
        DAPS(graph_data, base_score_function, neigh_coef)


def test_daps_call_without_labels(graph_data, base_score_function):
    daps = DAPS(graph_data, base_score_function, neigh_coef=0.5)
    logits = torch.tensor([
        [1.0, 0.5],
        [0.2, 0.8],
        [0.4, 0.6]
    ], dtype=torch.float32)
    
    base_scores = base_score_function(logits)

    diffusion_scores = torch.tensor([
        [base_scores[1, 0], base_scores[1, 1]],
        [(base_scores[0, 0]+base_scores[2, 0])/2, (base_scores[0, 1]+base_scores[2, 1])/2],
        [base_scores[1, 0], base_scores[1, 1]]
    ], dtype=torch.float32)
    
    expected_scores = 0.5 * diffusion_scores + 0.5 * base_scores
    
    scores = daps(logits)
    assert torch.allclose(scores, expected_scores, atol=1e-5)


def test_daps_call_with_labels(graph_data, base_score_function):
    daps = DAPS(graph_data, base_score_function, neigh_coef=0.5)
    
    logits = torch.tensor([
        [1.0, 0.5],
        [0.2, 0.8],
        [0.4, 0.6]
    ], dtype=torch.float32)
    
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    
    scores = daps(logits, labels)
    expected_scores = daps(logits)
    assert torch.allclose(scores, expected_scores[torch.arange(3), labels], atol=1e-5)
