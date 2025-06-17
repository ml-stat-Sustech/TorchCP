# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans

from torchcp.classification.predictor import ClusteredPredictor
from torchcp.classification.score import LAC
from torchcp.utils.common import DimensionError


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
    return LAC(score_type="softmax")


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


def test_invalid_split_initialization(mock_score_function, mock_model):
    with pytest.raises(ValueError, match="split should be one of 'proportional', 'doubledip', or 'random'."):
        ClusteredPredictor(mock_score_function, mock_model, split="error")


@pytest.mark.parametrize("alpha, ratio_clustering, num_clusters",
                         [(0.01, "auto", "auto"),
                          (0.05, "auto", "auto"),
                          (0.05, 0.5, "auto"),
                          (0.01, "auto", 2),
                          (0.01, 0.5, 2)])
def test_ratio_and_num_cluster_calculate_threshold(mock_score_function, mock_model, alpha, ratio_clustering,
                                                   num_clusters):
    n_z, n_o, n_t = 10, 95, 200
    logits = torch.randn(n_z + n_o + n_t, 3)
    zeros = torch.zeros(n_z, dtype=torch.long)
    ones = torch.ones(n_o, dtype=torch.long)
    twos = torch.full((n_t,), 2, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos])

    predictor = ClusteredPredictor(mock_score_function, mock_model, 1.0, 0.1, ratio_clustering, num_clusters)
    predictor.calculate_threshold(logits, labels, None)

    predictor = ClusteredPredictor(mock_score_function, mock_model, 1.0, 0.1, ratio_clustering, num_clusters)
    predictor.calculate_threshold(logits, labels, alpha)

    n_thresh = predictor._ClusteredPredictor__get_quantile_minimum(torch.tensor(alpha))
    n_min = torch.maximum(torch.tensor(n_z), n_thresh)
    num_remaining_classes = (n_z >= n_min).int() + (n_o >= n_min).int() + (n_t >= n_min).int()
    n_clustering = (n_min * num_remaining_classes / (75 + num_remaining_classes)).to(torch.int32)

    if num_clusters == "auto":
        assert predictor._ClusteredPredictor__num_clusters == torch.floor(n_clustering / 2).to(torch.int32)
    else:
        assert predictor._ClusteredPredictor__num_clusters == torch.tensor(num_clusters, dtype=torch.int32)
    if ratio_clustering == "auto":
        assert torch.allclose(predictor._ClusteredPredictor__ratio_clustering, n_clustering / n_min)
    else:
        assert torch.allclose(torch.tensor(predictor._ClusteredPredictor__ratio_clustering, dtype=torch.float32),
                              torch.tensor(ratio_clustering, dtype=torch.float32))


def test_clustering_calculate_threshold(mock_score_function, mock_model):
    # first situation
    alpha, ratio_clustering, num_clusters = 0.01, "auto", "auto"
    n_z, n_o = 70, 300
    logits = torch.randn(n_z + n_o, 2)
    zeros = torch.zeros(n_z, dtype=torch.long)
    ones = torch.ones(n_o, dtype=torch.long)
    labels = torch.cat([zeros, ones])

    predictor = ClusteredPredictor(mock_score_function, mock_model, 1.0, 0.1, ratio_clustering, num_clusters)
    predictor.calculate_threshold(logits, labels, alpha)

    assert torch.equal(predictor.cluster_assignments, torch.tensor([-1, -1], dtype=torch.int32))

    # second situation
    alpha, ratio_clustering, num_clusters = 0.01, "auto", "auto"
    n0, n1, n2, n3, n4 = 50, 100, 100, 100, 100
    logits = torch.randn(n0 + n1 + n2 + n3 + n4, 5)
    zeros = torch.zeros(n0, dtype=torch.long)
    ones = torch.ones(n1, dtype=torch.long)
    twos = torch.full((n2,), 2, dtype=torch.long)
    threes = torch.full((n3,), 3, dtype=torch.long)
    fours = torch.full((n4,), 4, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos, threes, fours])

    predictor = ClusteredPredictor(mock_score_function, mock_model, 1.0, 0.1, ratio_clustering, num_clusters,
                                   split="doubledip")
    predictor.calculate_threshold(logits, labels, alpha)

    scores = mock_score_function(logits, labels)
    filtered_scores = scores[n0:]
    filtered_labels = labels.clone()[n0:] - 1
    embeddings, class_cts = predictor._ClusteredPredictor__embed_all_classes(filtered_scores, filtered_labels)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=2023).fit(X=embeddings.numpy(),
                                                                    sample_weight=np.sqrt(class_cts.numpy()))
    nonrare_class_cluster_assignments = torch.tensor(kmeans.labels_)
    cluster_assignments = - torch.ones((5,), dtype=torch.int32)
    for cls, remapped_cls in {1: 0, 2: 1, 3: 2, 4: 3}.items():
        cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]

    assert torch.equal(predictor.cluster_assignments, cluster_assignments)

    result = predictor._ClusteredPredictor__compute_cluster_specific_qhats(cluster_assignments, scores, labels, alpha)
    assert torch.allclose(predictor.q_hat, result)


@pytest.mark.parametrize("split", ['proportional', 'doubledip', 'random'])
def test_split_data(mock_score_function, mock_model, split):
    logits = torch.randn(120, 3)
    zeros = torch.zeros(40, dtype=torch.long)
    ones = torch.ones(40, dtype=torch.long)
    twos = torch.full((40,), 2, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos])

    num_classes = logits.shape[1]
    classes_statistics = torch.tensor([torch.sum(labels == k).item() for k in range(num_classes)])
    scores = mock_score_function(logits, labels)

    torch.manual_seed(42)
    predictor = ClusteredPredictor(mock_score_function, mock_model, ratio_clustering=0.5, split=split)
    results = predictor._ClusteredPredictor__split_data(scores, labels, classes_statistics)

    torch.manual_seed(42)
    if split == 'proportional':
        classes_statistics = torch.tensor([40, 40, 40])
        n_k = torch.tensor([20, 20, 20])
        idx1 = torch.zeros(labels.shape, dtype=torch.bool)
        for k in range(3):
            idx = torch.argwhere(labels == k).flatten()
            random_indices = torch.randint(0, classes_statistics[k], (n_k[k],))
            selected_idx = idx[random_indices]
            idx1[selected_idx] = 1
        clustering_scores = scores[idx1]
        clustering_labels = labels[idx1]
        cal_scores = scores[~idx1]
        cal_labels = labels[~idx1]
    elif split == 'doubledip':
        clustering_scores, clustering_labels = scores, labels
        cal_scores, cal_labels = scores, labels
        idx1 = torch.ones((scores.shape[0])).bool()
    elif split == 'random':
        idx1 = torch.rand(size=(len(labels),)) < 0.5
        clustering_scores = scores[idx1]
        clustering_labels = labels[idx1]
        cal_scores = scores[~idx1]
        cal_labels = labels[~idx1]

    assert torch.equal(results[0], clustering_scores)
    assert torch.equal(results[1], clustering_labels)
    assert torch.equal(results[2], cal_scores)
    assert torch.equal(results[3], cal_labels)
    assert torch.equal(predictor.idx1, idx1)


@pytest.mark.parametrize("alpha", [0.2, 0.15, 0.10, 0.05, 0.01])
def test_get_quantile_minimum(predictor, alpha):
    result = predictor._ClusteredPredictor__get_quantile_minimum(torch.tensor(alpha))

    n = torch.tensor(0)
    while torch.ceil((n + 1) * (1 - alpha) / n) > 1:
        n += 1
    assert result.item() == n.item()


@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
def test_get_rare_classes(predictor, alpha):
    zeros = torch.zeros(8, dtype=torch.long)
    ones = torch.ones(18, dtype=torch.long)
    twos = torch.full((98,), 2, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos])

    result = predictor._ClusteredPredictor__get_rare_classes(labels, torch.tensor(alpha), 4)

    if alpha == 0.1:
        assert torch.equal(result, torch.tensor([0, 3]))
    elif alpha == 0.05:
        assert torch.equal(result, torch.tensor([0, 1, 3]))
    elif alpha == 0.01:
        assert torch.equal(result, torch.tensor([0, 1, 2, 3]))


@pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
def test_remap_classes(predictor, alpha):
    zeros = torch.zeros(8, dtype=torch.long)
    ones = torch.ones(18, dtype=torch.long)
    twos = torch.full((98,), 2, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos])

    if alpha == 0.1:
        rare_classes = torch.tensor([0, 3])
        remaining_idx = torch.ones(labels.shape, dtype=torch.bool)
        remaining_idx[:8] = False
        remapped_labels = torch.zeros((116,), dtype=torch.int32)
        remapped_labels[18:] = 1
        remapping = {1: 0, 2: 1}
    elif alpha == 0.05:
        rare_classes = torch.tensor([0, 1, 3])
        remaining_idx = torch.ones(labels.shape, dtype=torch.bool)
        remaining_idx[:26] = False
        remapped_labels = torch.zeros((98,), dtype=torch.int32)
        remapping = {2: 0}
    elif alpha == 0.01:
        rare_classes = torch.tensor([0, 1, 2, 3])
        remaining_idx = torch.zeros(labels.shape, dtype=torch.bool)
        remapped_labels = torch.tensor([])
        remapping = {}

    result = predictor._ClusteredPredictor__remap_classes(labels, rare_classes)
    assert torch.equal(result[0], remaining_idx)
    assert torch.equal(result[1], remapped_labels)
    assert result[2] == remapping


def test_embed_all_classes(predictor):
    scores_all = torch.arange(603).float()
    zeros = torch.zeros(101, dtype=torch.long)
    ones = torch.ones(201, dtype=torch.long)
    twos = torch.full((301,), 2, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos])

    embeddings = torch.tensor([
        [50, 60, 70, 80, 90],
        [201, 221, 241, 261, 281],
        [452, 482, 512, 542, 572]], dtype=torch.float32)
    cts = torch.tensor([101, 201, 301])

    result = predictor._ClusteredPredictor__embed_all_classes(scores_all, labels)
    assert torch.allclose(result[0], embeddings)
    assert torch.equal(result[1], cts)

    scores_all = torch.randn((603, 2))
    with pytest.raises(DimensionError, match="Expected 1-dimension"):
        predictor._ClusteredPredictor__embed_all_classes(scores_all, labels)


@pytest.mark.parametrize("alpha", [0.10])
def test_compute_cluster_specific_qhats(predictor, alpha):
    cluster_assignments = torch.tensor([-1, 0, 1], dtype=torch.int32)
    cal_class_scores = torch.arange(400).float()
    zeros = torch.zeros(100, dtype=torch.long)
    ones = torch.ones(100, dtype=torch.long)
    twos = torch.full((200,), 2, dtype=torch.long)
    cal_true_labels = torch.cat([zeros, ones, twos])

    result = predictor._ClusteredPredictor__compute_cluster_specific_qhats(cluster_assignments, cal_class_scores,
                                                                           cal_true_labels, alpha)

    assert result[0] == torch.inf
    assert result[1] == torch.tensor(100 + int((1 - alpha) * 100))
    assert result[2] == torch.tensor(200 + int((1 - alpha) * 200))


@pytest.mark.parametrize("alpha", [0.10])
def test_compute_class_specific_qhats(predictor, alpha):
    cal_class_scores = torch.arange(700).float()
    zeros = torch.zeros(100, dtype=torch.long)
    ones = torch.ones(200, dtype=torch.long)
    twos = torch.full((300,), 2, dtype=torch.long)
    threes = torch.full((100,), -1, dtype=torch.long)
    cal_true_clusters = torch.cat([zeros, ones, twos, threes])
    num_clusters = 3

    result = predictor._ClusteredPredictor__compute_class_specific_qhats(cal_class_scores, cal_true_clusters,
                                                                         num_clusters, alpha)

    assert result[0] == torch.tensor(int((1 - alpha) * 100))
    assert result[1] == torch.tensor(100 + int((1 - alpha) * 200))
    assert result[2] == torch.tensor(300 + int((1 - alpha) * 300))
    assert result[3] == torch.inf
