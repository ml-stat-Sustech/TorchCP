import pytest
import torch
from torchcp.llm.utils.scoring import geometric, marginal, first_k, first_k_no_mask, max, sum

class TestScoring:
    def test_geometric(self):
        p = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        expected_output = -torch.cumsum(torch.log(torch.maximum(1 - p, torch.tensor(1e-8))), dim=-1)
        output = geometric(p)
        assert torch.allclose(output, expected_output)

    def test_marginal(self):
        p = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        shifted = torch.pad(torch.maximum(1 - p, torch.tensor(1e-8)), ((0, 0), (1, 0)), constant_values=1)[:, :-1]
        expected_output = -torch.log(torch.maximum(1 - p, torch.tensor(1e-8))) - torch.cumsum(torch.log(shifted), dim=-1)
        output = marginal(p)
        assert torch.allclose(output, expected_output)

    def test_first_k(self):
        X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_output = torch.cumsum(torch.ones_like(X), dim=-1)
        output = first_k(X)
        assert torch.allclose(output, expected_output)

    def test_first_k_no_mask(self):
        X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_output = torch.cumsum(torch.ones_like(X), dim=-1)
        output = first_k_no_mask(X)
        assert torch.allclose(output, expected_output)

    def test_max(self):
        X = torch.tensor([[1, 3, 2], [4, 2, 5]])
        expected_output, _ = torch.cummax(X, dim=-1)
        output = max(X)
        assert torch.allclose(output, expected_output)

    def test_sum(self):
        X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected_output = torch.cumsum(X, dim=-1)
        output = sum(X)
        assert torch.allclose(output, expected_output)

if __name__ == "__main__":
    pytest.main()