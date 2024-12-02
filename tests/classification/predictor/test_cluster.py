import pytest
import torch

from torchcp.classification.scores import THR
from torchcp.classification.predictors import ClusteredPredictor


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x
    return MockModel()


@pytest.fixture
def mock_score_function():
    return THR(score_type="softmax")


@pytest.fixture
def predictor(mock_score_function, mock_model):
    return ClusteredPredictor(mock_score_function, mock_model, 2.0)


def test_clustered_predictor_initialization(predictor, mock_score_function, mock_model):
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model
    assert not predictor._model.training
    assert predictor._device == next(mock_model.parameters()).device
    assert predictor._logits_transformation.temperature == 2.0


@pytest.mark.parametrize("ratio_clustering", [-0.1, -1.0, 0, 1])
def test_invalid_ratio_clustering_initialization(mock_score_function, mock_model, ratio_clustering):
    with pytest.raises(ValueError, match="ratio_clustering should be 'auto' or a value"):
        ClusteredPredictor(mock_score_function, mock_model, ratio_clustering=ratio_clustering)


@pytest.mark.parametrize("num_clusters", [1.1, -10])
def test_invalid_num_clusters_initialization(mock_score_function, mock_model, num_clusters):
    with pytest.raises(ValueError, match="num_clusters should be 'auto' or a positive integer."):
        ClusteredPredictor(mock_score_function, mock_model, num_clusters=num_clusters)


def test_invalid_num_clusters_initialization(mock_score_function, mock_model):
    with pytest.raises(ValueError, match="split should be one of 'proportional', 'doubledip', or 'random'."):
        ClusteredPredictor(mock_score_function, mock_model, split="error")


@pytest.mark.parametrize("alpha, ratio_clustering, num_clusters, split", 
                         [(0.5, "auto", "auto", "random"),
                          (0.1, "auto", "auto", "random"),
                          (0.1, 0.5, "auto", "random"),
                          (0.1, "auto", 2, "random"),
                          (0.1, 0.5, 2, "random"),
                          (0.1, "auto", "auto", "proportional"),
                          (0.1, "auto", "auto", "doubledip")])
def test_calculate_threshold(mock_score_function, mock_model, alpha, ratio_clustering, num_clusters, split):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100, ))

    predictor = ClusteredPredictor(mock_score_function, mock_model, 1.0, ratio_clustering, num_clusters, split)
    predictor.calculate_threshold(logits, labels, alpha)




@pytest.mark.parametrize("alpha", [0, 1, -0.1, 2])
def test_invalid_calibrate_alpha(predictor, alpha):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100, ))
    with pytest.raises(ValueError, match="alpha should be a value"):
        predictor.calculate_threshold(logits, labels, alpha)

#     logits = torch.randn(10, 5)
#     labels = torch.randint(0, 5, (10,))
#     alpha = 0.05
    
#     score_function = MagicMock(return_value=torch.randn(10))


#     predictor = ClusteredPredictor(score_function=score_function, model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')

#     predictor.calculate_threshold(logits, labels, alpha)
    
#     assert predictor.q_hat is not None
 
#     assert predictor.num_clusters is not None


# def test_get_quantile_minimum():
#     predictor = ClusteredPredictor(score_function=MagicMock(), model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')
#     alpha = torch.tensor(0.05)
    
#     n_min = predictor._ClusteredPredictor__get_quantile_minimum(alpha)
#     assert n_min > 0


# def test_split_data():
#     scores = torch.randn(100)
#     labels = torch.randint(0, 5, (100,))
#     classes_statistics = torch.tensor([20, 20, 20, 20, 20])
    
#     predictor = ClusteredPredictor(score_function=MagicMock(), model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')
    
#     clustering_scores, clustering_labels, cal_scores, cal_labels = predictor._ClusteredPredictor__split_data(scores, labels, classes_statistics)
    
#     assert len(clustering_scores) + len(cal_scores) == len(scores)
#     assert len(clustering_labels) + len(cal_labels) == len(labels)


# def test_get_rare_classes():
#     labels = torch.randint(0, 5, (100,))
#     alpha = torch.tensor(0.05)
#     num_classes = 5
    
#     predictor = ClusteredPredictor(score_function=MagicMock(), model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')
#     rare_classes = predictor._ClusteredPredictor__get_rare_classes(labels, alpha, num_classes)
    
#     assert len(rare_classes) >= 0

# def test_remap_classes():
#     labels = torch.randint(0, 5, (100,))
#     rare_classes = torch.tensor([0, 2])
    
#     predictor = ClusteredPredictor(score_function=MagicMock(), model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')
#     remaining_idx, remapped_labels, remapping = predictor._ClusteredPredictor__remap_classes(labels, rare_classes)
    
#     assert len(remapped_labels) == len(remaining_idx)
#     assert len(remapping) > 0


# def test_embed_all_classes():
#     scores_all = torch.randn(100)
#     labels = torch.randint(0, 5, (100,))
    
#     predictor = ClusteredPredictor(score_function=MagicMock(), model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')
#     embeddings, cts = predictor._ClusteredPredictor__embed_all_classes(scores_all, labels)
    
#     assert embeddings.shape[0] == len(torch.unique(labels))
#     assert cts.shape[0] == len(torch.unique(labels))


# def test_compute_cluster_specific_qhats():
#     cluster_assignments = torch.tensor([0, 1, 1, 2, 2])
#     cal_class_scores = torch.randn(5)
#     cal_true_labels = torch.tensor([0, 1, 1, 2, 2])
#     alpha = 0.05
    
#     predictor = ClusteredPredictor(score_function=MagicMock(), model=None, temperature=1, ratio_clustering="auto", num_clusters="auto", split='random')
#     qhats = predictor._ClusteredPredictor__compute_cluster_specific_qhats(cluster_assignments, cal_class_scores, cal_true_labels, alpha)
    
#     assert len(qhats) == 3
