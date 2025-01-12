# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torchsort import soft_sort, soft_rank

from torchcp.classification.loss import ConfLearnLoss
from torchcp.classification.loss.conflearn import UniformMatchingLoss


def test_UniformMatchingLoss():
    loss_fn = UniformMatchingLoss()
    x = torch.rand((100,))
    torch.manual_seed(0)
    loss = loss_fn(x)

    torch.manual_seed(0)
    x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=0.1)
    i_seq = torch.arange(1.0, 1.0 + 100, device=x.device) / 100
    except_loss = torch.max(torch.abs(i_seq - x_sorted))
    assert loss == except_loss

    x = torch.rand((0,))
    loss = loss_fn(x)
    assert loss == 0


def test_initialization():
    loss_fn = ConfLearnLoss()
    assert isinstance(loss_fn.layer_prob, torch.nn.Softmax)
    assert loss_fn.layer_prob.dim == 1
    assert isinstance(loss_fn.criterion_scores, UniformMatchingLoss)


def test_forward():
    loss_fn = ConfLearnLoss()
    output = torch.rand((100, 2))
    target = torch.randint(0, 2, (100,))
    Z_batch = torch.randint(0, 2, (100,))

    torch.manual_seed(0)
    loss_scores = loss_fn(output, target, Z_batch)

    torch.manual_seed(0)
    idx_z = torch.where(Z_batch == 1)[0]
    except_loss_scores = loss_fn.compute_loss(output[idx_z], target[idx_z]).float()

    assert loss_scores == except_loss_scores


def test_compute_loss():
    loss_fn = ConfLearnLoss()
    y_train_pred = torch.rand((100, 2))
    y_train_batch = torch.randint(0, 2, (100,))

    torch.manual_seed(0)
    train_loss_scores = loss_fn.compute_loss(y_train_pred, y_train_batch)

    torch.manual_seed(0)
    train_proba = torch.nn.Softmax(dim=1)(y_train_pred)
    train_scores = loss_fn._ConfLearnLoss__compute_scores_diff(train_proba, y_train_batch)
    except_train_loss_scores = UniformMatchingLoss()(train_scores)

    assert train_loss_scores == except_train_loss_scores


def test_compute_scores_diff():
    loss_fn = ConfLearnLoss()
    proba_values = torch.rand((100, 2))
    Y_values = torch.randint(0, 2, (100,))

    torch.manual_seed(0)
    scores_t = loss_fn._ConfLearnLoss__compute_scores_diff(proba_values, Y_values)

    torch.manual_seed(0)
    proba_values = proba_values + 1e-6 * torch.rand((100, 2), dtype=float)
    proba_values = proba_values / torch.sum(proba_values, 1)[:, None]
    ranks_array_t = soft_rank(-proba_values, regularization_strength=0.1) - 1
    prob_sort_t = -soft_sort(-proba_values, regularization_strength=0.1)
    Z_t = prob_sort_t.cumsum(dim=1)

    ranks_t = torch.gather(ranks_array_t, 1, Y_values.reshape(100, 1)).flatten()
    prob_cum_t = loss_fn._ConfLearnLoss__soft_indexing(Z_t, ranks_t)
    prob_final_t = loss_fn._ConfLearnLoss__soft_indexing(prob_sort_t, ranks_t)
    except_scores_t = 1.0 - prob_cum_t + prob_final_t * torch.rand(100, dtype=float)

    assert torch.equal(scores_t, except_scores_t)


def test_soft_indicator():
    loss_fn = ConfLearnLoss()
    x = torch.rand((100, 2))
    a = torch.rand((100, 2))

    torch.manual_seed(0)
    out = loss_fn._ConfLearnLoss__soft_indicator(x, a)

    torch.manual_seed(0)
    except_out = torch.sigmoid(50 * (x - a + 0.5)) - (torch.sigmoid(50 * (x - a - 0.5)))
    except_out = except_out / (torch.sigmoid(torch.tensor(50 * 0.5)) - torch.sigmoid(-torch.tensor(50 * 0.5)))

    assert torch.equal(out, except_out)


def test_soft_indexing():
    loss_fn = ConfLearnLoss()
    z = torch.rand((100, 2))
    rank = torch.randint(0, 2, (100,))

    torch.manual_seed(0)
    weight = loss_fn._ConfLearnLoss__soft_indexing(z, rank)

    torch.manual_seed(0)
    I = torch.tile(torch.arange(2), (100, 1))
    except_weight = loss_fn._ConfLearnLoss__soft_indicator(I.T, rank).T
    except_weight = (except_weight * z).sum(dim=1)

    assert torch.equal(weight, except_weight)
