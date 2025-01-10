# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.graph.predictor.base import BasePredictor
from torchcp.utils.common import calculate_conformal_value


class SplitPredictor(BasePredictor):
    """
    Method: Split Conformal Prediction (Vovk et a., 2005).
    Paper: Algorithmic Learning in a Random World
    Link: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    """

    def __init__(self, graph_data, score_function, model=None):
        super().__init__(graph_data, score_function, model)

    # The calibration process ########################################################

    def calibrate(self, cal_idx, alpha):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x,
                                 self._graph_data.edge_index)
        self.calculate_threshold(logits, cal_idx, self._label_mask, alpha)

    def calculate_threshold(self, logits, cal_idx, label_mask, alpha):
        """
        Calculate the conformal prediction threshold for a given calibration set.

        This method computes a threshold (`q_hat`) based on the non-conformity scores 
        of the calibration set. The threshold ensures that the conformal prediction 
        meets the desired significance level (`alpha`).

        Args:
            logits (torch.Tensor):
                The raw model outputs (logits) for all samples in the dataset. 
                Shape: [num_samples, num_classes].
            cal_idx (torch.Tensor or list): 
                Indices specifying the samples in the calibration set. 
                Shape: [num_calibration_samples].
            label_mask (torch.Tensor): 
                A boolean tensor indicating the presence of valid labels for each 
                sample and class. Shape: [num_samples, num_classes].
            alpha (float): 
                The significance level, a value in the range (0, 1), representing 
                the acceptable error rate.
        """
        logits = logits.to(self._device)
        label_mask = label_mask.to(self._device)

        scores = self.score_function(logits)
        cal_scores = scores[cal_idx][label_mask[cal_idx]]
        self.q_hat = self._calculate_conformal_value(cal_scores, alpha)

    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value(scores, alpha, marginal_q_hat)

    # The prediction process ########################################################

    def predict(self, eval_idx):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x,
                                 self._graph_data.edge_index)
        sets = self.predict_with_logits(logits, eval_idx)
        return sets

    def predict_with_logits(self, logits, eval_idx, q_hat=None):
        """
        Generate prediction sets based on the logits and the conformal threshold.

        This method constructs prediction sets by comparing the non-conformity scores 
        (calculated from the logits) to a predefined threshold (`q_hat`). If `q_hat` 
        is not provided, it defaults to the value of `self.q_hat`, which should have been 
        set during the calibration phase.

        Args:
            logits (torch.Tensor): 
                The raw output of the model (before applying softmax).
                Shape: [num_samples, num_classes].

            eval_idx (torch.Tensor or list): 
                Indices of the samples in the evaluation or test set. 
                Shape: [num_test_samples].

            q_hat (float, optional): 
                The conformal threshold used to generate prediction sets. If not provided, 
                `self.q_hat` (calculated during the calibration phase) will be used.

        Returns:
            list:
                A list containing prediction sets for the test samples, depending on 
                the specific conformal prediction method implemented.
        """
        logits = logits.to(self._device)
        scores = self.score_function(logits)

        eval_scores = scores[eval_idx]
        if q_hat is None:
            if not hasattr(self, "q_hat"):
                raise ValueError(
                    "Ensure self.q_hat is not None. Please perform calibration first.")
            q_hat = self.q_hat

        S = self._generate_prediction_set(eval_scores, q_hat)
        return S

    # The evaluation process ########################################################

    def evaluate(self, eval_idx):
        """
        Evaluate the model's conformal prediction performance on a given evaluation set.

        This method performs evaluation by first making predictions using the model's raw outputs 
        (logits) and then calculating several performance metrics based on the prediction sets 
        generated for the evaluation samples. It calculates the coverage rate, average prediction set 
        size, and singleton hit ratio, and returns these metrics as a dictionary.

        Parameters:
            eval_idx (torch.Tensor or list): 
                Indices of the samples in the evaluation or test set. 
                Shape: [num_test_samples].

        Returns:
            dict:
                A dictionary containing the evaluation results. The dictionary includes:
                - "Coverage_rate": The proportion of test samples for which the true label is included 
                in the prediction set.
                - "Average_size": The average size of the prediction sets.
                - "Singleton_hit_ratio": The ratio of singleton (i.e., single-class) prediction sets 
                where the predicted class matches the true label.
        """
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x,
                                 self._graph_data.edge_index)
        prediction_sets = self.predict_with_logits(logits, eval_idx)

        res_dict = {"coverage_rate": self._metric('coverage_rate')(prediction_sets, self._graph_data.y[eval_idx]),
                    "average_size": self._metric('average_size')(prediction_sets, self._graph_data.y[eval_idx]),
                    "singleton_hit_ratio": self._metric('singleton_hit_ratio')(prediction_sets,
                                                                               self._graph_data.y[eval_idx])}
        return res_dict
