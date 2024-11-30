import pytest

import torch
from torchcp.classification.scores import THR
from torchcp.graph.scores import DAPS


@pytest.fixture
def graph_data():
    adj = torch.tensor([[0, 1, 1, 2],
                        [1, 0, 2, 1]], dtype=torch.int)
    degs = torch.tensor([1, 2, 1], dtype=torch.int)
    return {"adj": adj, "degs": degs}


@pytest.fixture
def base_score_function():
    return THR(score_type="softmax")


def test_daps_initialization(graph_data, base_score_function):
    daps = DAPS(graph_data, base_score_function, neigh_coef=0.7)
    assert daps._neigh_coef == 0.7
    assert daps._adj.equal(graph_data["adj"])
    assert daps._degs.equal(graph_data["degs"])


# # 测试非法的 neigh_coef 参数
# @pytest.mark.parametrize("neigh_coef", [-0.1, 1.1, -5, 1.5])
# def test_invalid_neigh_coef(graph_data, base_score_function, neigh_coef):
#     with pytest.raises(ValueError, match="The parameter 'neigh_coef' must be a value between 0 and 1."):
#         DAPS(graph_data, base_score_function, neigh_coef)


# # 测试 __call__ 方法（无标签）
# def test_daps_call_without_labels(graph_data, base_score_function):
#     daps = DAPS(graph_data, base_score_function, neigh_coef=0.5)
    
#     logits = torch.tensor([
#         [1.0, 0.5],
#         [0.2, 0.8],
#         [0.4, 0.6]
#     ], dtype=torch.float32)
    
#     expected_base_scores = logits
#     expected_diffusion_scores = torch.tensor([
#         [0.1, 0.4],
#         [0.7, 0.55],
#         [0.1, 0.4]
#     ], dtype=torch.float32)  # 根据邻接矩阵和度数计算的扩散分数
    
#     expected_scores = 0.5 * expected_diffusion_scores + 0.5 * expected_base_scores
    
#     scores = daps(logits)
#     assert torch.allclose(scores, expected_scores, atol=1e-5)


# # 测试 __call__ 方法（带标签）
# def test_daps_call_with_labels(graph_data, base_score_function):
#     daps = DAPS(graph_data, base_score_function, neigh_coef=0.5)
    
#     logits = torch.tensor([
#         [1.0, 0.5],
#         [0.2, 0.8],
#         [0.4, 0.6]
#     ], dtype=torch.float32)
    
#     labels = torch.tensor([0, 1, 1], dtype=torch.long)  # 节点对应的标签
    
#     scores = daps(logits, labels)
#     expected_scores = daps(logits)
#     assert torch.allclose(scores, expected_scores[torch.arange(3), labels], atol=1e-5)
