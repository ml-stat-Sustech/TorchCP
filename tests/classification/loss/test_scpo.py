# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import pytest

from torchcp.classification.score import THR
from torchcp.classification.predictor import SplitPredictor as Predictor
from torchcp.classification.loss.scpo import SCPOLoss
from torchcp.classification.loss.conftr import ConfTrLoss

def test_scpo_init_valid_params():
    predictor = Predictor(THR())
    scpo = SCPOLoss(predictor=predictor, alpha=0.05)
    assert scpo.predictor == predictor
    assert scpo.alpha == 0.05
    assert scpo.lambda_val == 500
    assert scpo.transform == torch.log
    assert type(scpo.size_loss_fn) is ConfTrLoss
    assert type(scpo.coverage_loss_fn) is ConfTrLoss

    scpo = SCPOLoss(predictor=predictor, alpha=0.05, loss_transform='neg_inv')
    assert scpo.transform(2) == -0.5
    assert scpo.transform(4) == -0.25


def test_scpo_init_invalid_fraction():
    predictor = Predictor(THR())
    with pytest.raises(ValueError, match="loss_transform should be log or neg_inv"):
        SCPOLoss(predictor=predictor, alpha=0.05, loss_transform="square")


@pytest.fixture
def sample_inputs():
    logits = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9]])
    labels = torch.tensor([2, 1, 2])
    return logits, labels


def test_forward(sample_inputs):
    logits, labels = sample_inputs
    scpo = SCPOLoss(predictor=Predictor(THR()), alpha=0.1)

    results = scpo(logits, labels)

    test_scores = 1 - torch.softmax(logits, dim=1)
    excepted_results = scpo.compute_loss(test_scores, labels, 1)
    assert torch.allclose(results, excepted_results)


def test_compute_loss(sample_inputs):
    test_scores, test_labels = sample_inputs
    scpo = SCPOLoss(predictor=Predictor(THR()), alpha=0.1)
    results = scpo.compute_loss(test_scores, test_labels, 1)

    size_loss = scpo.size_loss_fn.compute_loss(test_scores, test_labels, 1)
    coverage_loss = scpo.coverage_loss_fn.compute_loss(test_scores, test_labels, 1)
    excepted_results = torch.log(size_loss + 500 * coverage_loss)
    assert results == excepted_results
