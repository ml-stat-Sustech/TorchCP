# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch

from torchcp.regression.predictor.split import SplitPredictor
from torchcp.selection.utils.metrics import Metrics


class ConformalSelector(SplitPredictor):
    """
    Conformal Selection:
        a screening procedure that aims to select candidates whose unobserved outcomes exceed user-specified value.

    Args:
        score_function (torchcp.regression.scores): A class that implements the score function.
        model (torch.nn.Module): A PyTorch model capable of outputting quantile values.
            The model should be an initialization model that has not been trained.
        alpha (float, optional): The significance level. Default is 0.1.
        device (torch.device, optional): The device on which the model is located. Default is None.

    Reference:
        Paper: Selection by Prediction with Conformal p-values (Jin et al., 2023)
        Link: https://arxiv.org/pdf/2210.01408
        Github: https://github.com/ying531/conformal-selection
    """

    def __init__(self, score_function, testing_correction, model, alpha=0.1, device=None):
        super().__init__(score_function, model, alpha, device)
        self.testing_correction = testing_correction
        self._metric = Metrics()


    def calibrate(self, cal_dataloader):
        self._model.eval()
        predicts_list, y_truth_list = [], []
        with torch.no_grad():
            for tmp_x, tmp_labels in cal_dataloader:
                tmp_x, tmp_labels = tmp_x.to(self._device), tmp_labels.to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)

        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)
        self.cal_scores = self.score_function(predicts, y_truth)


    def select(self, data_loader, thresholds):
        """
        Evaluate the performance of conformal selection on a test dataset by calculating false discovery proportion
        (FDP) and power of the selection set.

        Args:
            data_loader (DataLoader): The DataLoader providing the test data batches.
            thresholds (torch.Tensor): A tensor of user-defined thresholds.

        Returns:
            dict: A dictionary containing:
                - "False discovery proportion": The FDP of the selection set.
                - "Power": The power of the selection set.

        Example::

            >>> eval_results = selector.evaluate(test_loader, thresholds)
            >>> print(eval_results)
        """
        self._model.eval()
        y_truth_list = []
        predicts_list = []
        with torch.no_grad():
            for examples in data_loader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)
        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)
        scores = self.score_function(predicts, thresholds)

        n_cal, n_test = self.cal_scores.shape[0], scores.shape[0]

        # Compute p-values with tie-breaking
        u = torch.rand(n_test)
        count_less = (self.cal_scores.view(1, n_cal) < scores.view(n_test, 1)).sum(dim=1)
        count_tie = (self.cal_scores.view(1, n_cal) == scores.view(n_test, 1)).sum(dim=1) + 1
        p_values = (count_less + count_tie * u) / (n_cal + 1)

        indices = self.testing_correction(p_values, self.alpha)

        # Evaluation
        res_dict = {"false_discovery_proportion": self._metric("false_discovery_proportion")(y_truth, thresholds,
                                                                                             indices),
                    "power": self._metric("power")(y_truth, thresholds, indices)}
        return res_dict
