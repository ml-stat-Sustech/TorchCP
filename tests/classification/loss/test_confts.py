# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.classification.loss.confts import ConfTSLoss
from torchcp.classification.predictor import SplitPredictor as Predictor
from torchcp.classification.score import THR


def test_confts_init_valid_params():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    assert confts.predictor == predictor
    assert confts.alpha == 0.05
    assert confts.fraction == 0.2
    assert confts.soft_qunatile is True


def test_confts_init_invalid_fraction():
    predictor = Predictor(THR())
    with pytest.raises(ValueError, match="fraction should be a value in \\(0,1\\)."):
        ConfTSLoss(predictor=predictor, alpha=0.05, fraction=1.2)


def test_confts_init_invalid_alpha():
    predictor = Predictor(THR())
    with pytest.raises(ValueError, match="alpha should be a value in \\(0,1\\)."):
        ConfTSLoss(predictor=predictor, alpha=1.2, fraction=0.2)


def test_confts_forward():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    loss = confts.forward(logits, labels)
    assert loss is not None


def test_confts_compute_loss():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    test_scores = torch.randn(80, 10)
    test_labels = torch.randint(0, 10, (80,))
    tau = torch.tensor(0.5)
    loss = confts.compute_loss(test_scores, test_labels, tau)
    assert loss is not None


def test_confts_neural_sort():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    scores = torch.randn(10)
    P_hat = confts._ConfTSLoss__neural_sort(scores)
    assert P_hat.shape == (10, 10)


def test_confts_soft_quantile():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    scores = torch.randn(10)
    quantile = confts._soft_quantile(scores, 0.5)
    assert quantile.shape == torch.Size([])


def test_confts_forward_with_non_soft_quantile():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2, soft_qunatile=False)
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    loss = confts.forward(logits, labels)
    assert loss is not None


def test_confts_soft_quantile_with_dim_permutation():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    scores = torch.randn(3, 4, 5)
    quantile = confts._soft_quantile(scores, 0.5, dim=1)
    assert quantile.shape == torch.Size([3, 5])


def test_confts_soft_quantile_with_multiple_quantiles():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    scores = torch.randn(10)
    quantiles = confts._soft_quantile(scores, [0.25, 0.5, 0.75])
    assert quantiles.shape == torch.Size([3])


def test_confts_soft_quantile_with_dim_permutation_and_multiple_quantiles():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    scores = torch.randn(3, 4, 5)
    quantiles = confts._soft_quantile(scores, [0.25, 0.5, 0.75], dim=1)
    assert quantiles.shape == torch.Size([3, 3, 5])


def test_confts_soft_quantile_with_dim_permutation_and_single_quantile():
    predictor = Predictor(THR())
    confts = ConfTSLoss(predictor=predictor, alpha=0.05, fraction=0.2)
    scores = torch.randn(3, 4, 5)
    quantile = confts._soft_quantile(scores, 0.5, dim=1)
    assert quantile.shape == torch.Size([3, 5])
