import pytest

import torch
from torch_geometric.data import Data

from torchcp.classification.scores import THR
from torchcp.graph.scores import SNAPS


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


@pytest.mark.parametrize("lambda_val", [-0.1, 1.1, -5, 1.5])
@pytest.mark.parametrize("mu_val", [-0.1, 1.1, -5, 1.5])
def test_invalid_lambda_mu_values(graph_data, base_score_function, lambda_val, mu_val):
    with pytest.raises(ValueError, match="The parameter 'lambda_val' must be a value between 0 and 1."):
        SNAPS(graph_data, base_score_function, lambda_val=lambda_val)

    with pytest.raises(ValueError, match="The parameter 'mu_val' must be a value between 0 and 1."):
        SNAPS(graph_data, base_score_function, mu_val=mu_val)

    with pytest.raises(ValueError, match="The summation of 'lambda_val' and 'mu_val' must not be greater than 1."):
        SNAPS(graph_data, base_score_function, lambda_val=0.6, mu_val=0.6)