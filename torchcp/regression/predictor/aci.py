# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import math
import warnings
from tqdm import tqdm

from torchcp.regression.predictor.split import SplitPredictor


class ACIPredictor(SplitPredictor):
    """
    Adaptive Conformal Inference.
    
    methods for forming prediction sets in an online setting where the data generating 
    distribution is allowed to vary over time in an unknown fashion.
    
    Args:
        score_function (torchcp.regression.scores): A class that implements the score function.
        model (torch.nn.Module): A PyTorch model capable of outputting quantile values. 
            The model should be an initialization model that has not been trained.
        gamma (float): Step size parameter for adaptive adjustment of alpha. Must be greater than 0.
        
    Reference:  
        Paper: Adaptive conformal inference Under Distribution Shift (Gibbs et al., 2021)
        Link: https://arxiv.org/abs/2106.00170
        Github: https://github.com/isgibbs/AdaptiveConformal
        
    """

    def __init__(self, score_function, model, gamma):
        super().__init__(score_function, model)
        if gamma <= 0:
            raise ValueError("gamma must be greater than 0.")

        self.gamma = gamma
        self.alpha_t = None
        self.model_backbone = model
        self.train_indicate = False

    def train(self, train_dataloader, alpha, **kwargs):
        """
        Train and calibrate the predictor using the training data.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            alpha (float): Desired initial coverage rate.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): The model to be trained.
                - epochs (int, optional): Number of training epochs.
                - criterion (nn.Module, optional): Loss function.
                - lr (float, optional): Learning rate for the optimizer.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training.
                - verbose (bool, optional): If True, prints training progress.
            
        .. note::
            This function is necessary for ACI predictor. 
            We provide a default training method, and users can change the hyperparameters :attr:`kwargs` to modify the training process.
            If the user wants to fully control the training process, it can be achieved by rewriting the :func:`train` of the score function.
        """
        super().train(train_dataloader, alpha=alpha, **kwargs)
        super().calibrate(train_dataloader, alpha)
        self.alpha = alpha
        self.alpha_t = alpha
        self.train_dataloader = train_dataloader
        self.train_indicate = True

    def calculate_err_rate(self, x_batch, y_batch_last, pred_interval_last):
        """
        Calculate the error rate for the previous prediction intervals.

        Args:
            x_batch (torch.Tensor): Input features for the current batch.
            y_batch_last (torch.Tensor): True labels from the previous step.
            pred_interval_last (torch.Tensor): Prediction intervals from the previous step.

        Returns:
            float: Weighted error rate based on historical predictions.
        """
        steps_t = len(y_batch_last)
        w_s = (steps_t - torch.arange(steps_t)).to(self._device)
        w_s = torch.pow(0.95, w_s)
        w_s = w_s / torch.sum(w_s)
        err = x_batch.new_zeros(steps_t, self.q_hat.shape[0])
        err = ((y_batch_last >= pred_interval_last[..., 0, 1]) | (y_batch_last <= pred_interval_last[..., 0, 0])).int()
        err_t = torch.sum(w_s * err)
        return err_t
    
    def predict(self, x_batch, x_lookback=None, y_lookback=None, pred_interval_lookback=None, train=False, update_alpha=False):
        """
        Generates conformal prediction intervals for a given batch of input data. 
        This function can also optionally retrain the model or update the conformal 
        score threshold (alpha) based on historical data.

        Args:
            x_batch (Tensor): A batch of input features for which predictions and 
                prediction intervals are to be generated. Shape depends on the model's 
                input requirements (e.g., [batch_size, num_features]).
                
            x_lookback (Tensor, optional): Historical input features used for retraining 
                or updating model calibration. If provided, `y_lookback` must also be provided. 
                Default is `None`.
                
            y_lookback (Tensor, optional): Historical target values corresponding to 
                `x_lookback`. Used for retraining or updating model calibration. If provided, 
                `x_lookback` must also be provided. Default is `None`.
                
            pred_interval_lookback (Tensor, optional): Previously generated prediction intervals 
                that can be used for calibration. If not provided, prediction intervals will 
                be computed using the model's predictions and the current quantile value `q_hat`.
                Default is `None`.

            train (bool, optional): Whether to retrain the model using the `x_lookback` and 
                `y_lookback` data. If `True`, both `x_lookback` and `y_lookback` must be provided. 
                If `False`, the model will not be retrained. Default is `False`.

            update_alpha (bool, optional): Whether to update the conformal score threshold (`alpha`)
                based on the error rate observed in the prediction intervals. If `True`, both 
                `x_lookback` and `y_lookback` must be provided. Default is `False`.

        Returns:
            Prediction intervals (Tensor): The conformal prediction intervals for the input batch `x_batch`
        
        Raises:
            ValueError: If `x_lookback` is provided but `y_lookback` is not, or vice versa. 
                Both must be provided together or both must be `None`.

        Notes:
            - If `train` is set to `True` but `x_lookback` and `y_lookback` are not provided, 
            the function will issue a warning and skip retraining.
            - If `update_alpha` is set to `True` but `x_lookback` and `y_lookback` are not provided, 
            the function will use the current value of `alpha` instead of recalibrating it.
            - The conformal score threshold (`alpha`) is updated using a time-decayed approach, 
            where the rate of adjustment depends on the parameter `gamma`.
        """
        if self.train_indicate is False:
            raise ValueError("The predict function must be called after the train function is called")
        self._model.eval()
        
        if (x_lookback is None) != (y_lookback is None):
            raise ValueError("x_lookback, y_lookback must either be provided or be None.")
        
        if x_lookback is not None:
            x_lookback = x_lookback.to(self._device)
        if y_lookback is not None:
            y_lookback = y_lookback.to(self._device)
        
        if (x_lookback is not None) and (y_lookback is not None) and (pred_interval_lookback is None):
            predicts_batch = self._model(x_batch.to(self._device)).float()
            pred_interval_lookback = self.generate_intervals(predicts_batch, self.q_hat)
        
        if train == True:
            if (x_lookback is not None) and (y_lookback is not None):
                back_dataset = torch.utils.data.TensorDataset(x_lookback, y_lookback)
                back_dataloader = torch.utils.data.DataLoader(back_dataset, batch_size=min(self.train_dataloader.batch_size, 
                                                                    math.floor(len(x_lookback)/2)), shuffle=False)
                self._model = self.score_function.train(back_dataloader, model=self.model_backbone, alpha= self.alpha, device=self._device, verbose=False)
            else:
                warnings.warn("Training is enabled but x_lookback and y_lookback are not provided. The model will not be retrained.", UserWarning)
        
        if update_alpha == True:
            if (x_lookback is not None) and (y_lookback is not None):
                err_t = self.calculate_err_rate(x_batch, y_lookback, pred_interval_lookback)
                self.scores = self.calculate_score(self._model(x_lookback).float(), y_lookback)
            else:
                err_t = self.alpha

            self.alpha_t = max(1/(self.scores.shape[0]+1), min(0.9999, self.alpha_t + self.gamma * (self.alpha - err_t)))
            self.q_hat = self._calculate_conformal_value(self.scores, self.alpha_t)
        
        predicts_batch = self._model(x_batch.to(self._device)).float()
        return self.generate_intervals(predicts_batch, self.q_hat)
        
    def evaluate(self, data_loader, lookback=200, retrain_gap=1, update_alpha_gap=1):
        """
        Evaluate the model using a test dataset and compute performance metrics such as 
        coverage rate and average prediction interval size. The evaluation process 
        optionally includes periodic retraining of the model and updating of the conformal 
        score threshold (`alpha`).

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset. 
                It should provide batches of input features and ground-truth labels for testing.
            
            lookback (int, optional): The number of historical data points (from the training dataset) 
                to use for initializing lookback buffers (`x_lookback`, `y_lookback`, and 
                `pred_interval_lookback`). This value must be less than or equal to the size of the 
                training dataset. Default is 200.

            retrain_gap (int, optional): The interval (in terms of number of test samples processed) 
                at which the model is retrained using the lookback data. If set to 0, retraining is 
                disabled. Default is 1.

            update_alpha_gap (int, optional): The interval (in terms of number of test samples processed) 
                at which the conformal score threshold (`alpha`) is updated using the lookback data. 
                If set to 0, updating `alpha` is disabled. Default is 1.

        Returns:
            dict: A dictionary containing the following metrics:
                - "Coverage_rate" (float): The proportion of true labels that fall within the 
                predicted intervals.
                - "Average_size" (float): The average size of the prediction intervals.

        Raises:
            ValueError: If `lookback` is greater than the size of the training dataset.

        Notes:
            - The lookback buffers (`x_lookback`, `y_lookback`, and `pred_interval_lookback`) are 
            initialized using the last `lookback` samples from the training dataset.
            - During the evaluation process:
                - If `retrain_gap` > 0, the model is retrained periodically (every `retrain_gap` 
                samples) using the lookback buffers.
                - If `update_alpha_gap` > 0, the conformal score threshold (`alpha`) is updated 
                periodically (every `update_alpha_gap` samples) based on the lookback buffers.
            - The lookback buffers are updated dynamically after processing each test sample 
            with the latest predictions, inputs, and ground-truth labels from the evaluation dataset.
        """
        train_dataset = self.train_dataloader.dataset
        if lookback > len(train_dataset):
            raise ValueError("lookback cannot be set above the length of train_dataloader")
        
        ts_dataloader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False, pin_memory=True)
        samples = [train_dataset[i] for i in range(len(train_dataset) - lookback, len(train_dataset))]

        x_lookback = torch.stack([sample[0] for sample in samples]).to(self._device)
        y_lookback = torch.stack([sample[1] for sample in samples]).to(self._device)
        pred_interval_lookback = self.predict(x_lookback)


        y_list, predict_list = [], []
        for idx, (x, y) in enumerate(tqdm(ts_dataloader, desc="Processing Evaluation")):
            x = x.to(self._device)
            y = y.to(self._device)
            if (retrain_gap != 0) and (idx % retrain_gap == 0) and (update_alpha_gap != 0) and (idx % update_alpha_gap == 0):
                pred_interval = self.predict(x_batch=x, x_lookback=x_lookback, y_lookback=y_lookback, 
                                             pred_interval_lookback=pred_interval_lookback, 
                                             train=True, update_alpha=True)
            elif (retrain_gap != 0) and (idx % retrain_gap == 0):
                pred_interval = self.predict(x_batch=x, x_lookback=x_lookback, y_lookback=y_lookback, 
                                             pred_interval_lookback=pred_interval_lookback, 
                                             train=True, update_alpha=False)
            elif (update_alpha_gap != 0) and (idx % update_alpha_gap == 0):
                pred_interval = self.predict(x_batch=x, x_lookback=x_lookback, y_lookback=y_lookback, 
                                             pred_interval_lookback=pred_interval_lookback, 
                                             train=False, update_alpha=True)
            else:
                pred_interval = self.predict(x_batch=x, x_lookback=x_lookback, y_lookback=y_lookback, 
                                             pred_interval_lookback=pred_interval_lookback, 
                                             train=False, update_alpha=False)
            y_list.append(y)
            predict_list.append(pred_interval)
            pred_interval_lookback = torch.cat([pred_interval_lookback, pred_interval], dim=0)[-lookback:]
            x_lookback = torch.cat([x_lookback, x], dim=0)[-lookback:]
            y_lookback = torch.cat([y_lookback, y], dim=0)[-lookback:]
            
        predicts = torch.cat(predict_list, dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)
        
        res_dict = {
            "coverage_rate": self._metric('coverage_rate')(predicts, test_y),
            "average_size": self._metric('average_size')(predicts)
        }
        return res_dict
