# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .cqr import CQR


class ACI(CQR):
    """
    Adaptive Conformal Inference.
    
    methods for forming prediction sets in an online setting where the data generating 
    distribution is allowed to vary over time in an unknown fashion.
    
    Args:
        model (torch.nn.Module): A PyTorch model capable of outputting quantile values.
        gamma (float): Step size parameter for adaptive adjustment of alpha. Must be greater than 0.
        
    Reference:  
        Paper: Adaptive conformal inference Under Distribution Shift (Gibbs et al., 2021)
        Link: https://arxiv.org/abs/2106.00170
        Github: https://github.com/isgibbs/AdaptiveConformal
        
    """

    def __init__(self, model, gamma):
        super().__init__(model)
        assert gamma > 0, "gamma must be greater than 0."
        self.__gamma = gamma
        self.alpha_t = None

    def calculate_threshold(self, predicts, y_truth, alpha):
        """
        Calculates the conformal threshold value `q_hat` based on predictions and true values,
        using the initial significance level :attr:`alpha`.

        Args:
            predicts (torch.Tensor): Predicted values from the model, of shape (batch_size, 2).
            y_truth (torch.Tensor): Ground truth values, of shape (batch_size,).
            alpha (float): Initial significance level for setting up the prediction intervals.
            
        .. Note::
            Procedure:
            1. Computes the scores based on the deviation of `predicts` from `y_truth`.
            2. Calculates the conformal quantile (q_hat) from the scores using the provided alpha.
            3. Stores alpha as `self.alpha` and initializes `self.alpha_t` for adaptive adjustment.

        """
        self.scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(self.scores, alpha)
        self.alpha = alpha
        self.alpha_t = alpha

    def predict(self, x, y_t=None, pred_interval_t=None):
        """
        Predicts the interval for the input features at time t+1, adapting based on 
        the previous intervals' performance.

        Args:
            x (torch.Tensor): Input features for prediction at time t+1, of shape (batch_size, input_dim).
            y_t (torch.Tensor, optional): True values at time t for error rate computation. Defaults to None.
            pred_interval_t (torch.Tensor, optional): Prediction intervals at time t. Required if :attr:`y_t` is provided.

        Returns:
            torch.Tensor: The prediction interval for each sample in `x`, of shape (batch_size, 2).

        .. Note::
            - If `y_t` is provided, it calculates the error rate as a weighted sum of recent error rates, 
            using a decay factor of 0.95. This allows for an adaptive approach to adjust the interval width.
            
            - Procedure:
                1. Sets the model to evaluation mode and transfers `x` to the device.
                2. Calculates the error rate `err_t` from previous intervals if `y_t` is provided. Otherwise, uses `self.alpha`.
                3. Adapts `alpha_t` by adjusting based on `err_t` and `self.alpha` using the step size `gamma`.
                4. Predicts the next quantiles and calculates prediction intervals based on updated `alpha_t`.

        """
        self._model.eval()
        x = x.to(self._device)

        #######################
        # Count the error rate in the previous steps
        #######################
        if y_t is None:
            err_t = self.alpha
        else:
            if y_t.dim() == 0:
                y_t = torch.tensor([y_t.item()]).to(self._device)
            if len(y_t.shape) == 0:
                err_t = ((y_t >= pred_interval_t[..., 0]) & (y_t <= pred_interval_t[..., 1])).int()
            else:
                steps_t = len(y_t)
                w = torch.arange(steps_t).to(self._device)
                w = torch.pow(0.95, w)
                w = w / torch.sum(w)
                err = x.new_zeros(steps_t, self.q_hat.shape[0])
                for i in range(steps_t):
                    err[i] = ((y_t >= pred_interval_t[..., 0]) & (y_t <= pred_interval_t[..., 1])).int()
                err_t = torch.sum(w * err)

        # Adaptive adjust the value of alpha
        self.alpha_t = self.alpha_t + self.__gamma * (self.alpha - err_t)
        predicts_batch = self._model(x.to(self._device)).float()
        if len(predicts_batch.shape) == 1:
            predicts_batch = predicts_batch.unsqueeze(0)
        q_hat = self._calculate_conformal_value(self.scores, self.alpha_t)
        prediction_intervals = x.new_zeros(self.q_hat.shape[0], 2)
        prediction_intervals[:, 0] = predicts_batch[:, 0] - q_hat.view(self.q_hat.shape[0], 1)
        prediction_intervals[:, 1] = predicts_batch[:, 1] + q_hat.view(self.q_hat.shape[0], 1)
        return prediction_intervals
