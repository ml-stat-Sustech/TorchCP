# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import torch

from torchcp.utils.common import get_device
from ..utils.metrics import Metrics
from .split import SplitPredictor


class EnsemblePredictor(SplitPredictor):
    """
    Ensemble Conformal Prediction Interval

    There are two main implementations for `score_function`:
    
    1. EnbPI:
        - Paper: Conformal Prediction Interval for Dynamic Time-Series (Xu and Xie, 2020)
        - Link: https://proceedings.mlr.press/v139/xu21h.html
        - Github: https://github.com/hamrel-cxu/EnbPI
    
    2. EnCQR:
        - Paper: Ensemble Conformalized Quantile Regression for Probabilistic 
                Time Series Forecasting (Jensen et al., 2022)
        - Link: https://ieeexplore.ieee.org/abstract/document/9940232
        - Github: https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression
       
    Args:
        model (torch.nn.Module): The base model to be used in the ensemble.
        score_function (torchcp.regression.scores): The method for calculating scores and prediction intervals.
                        Can be :class:`torchcp.regression.scores.split` for EnbPI, 
                        :class:`torchcp.regression.scores.CQR` for EnCQR, or other variants.
        aggregation_function (str or callable): A function to aggregate predictions from multiple 
                        models in the ensemble. Options include:
                        - torch.mean: Computes the mean of the predictions.
                        - torch.median: Computes the median of the predictions.
                        - Custom function: Should accept a tensor and dimension as input, returning the result.
    """
    
    def __init__(self, model, score_function, aggregation_function='mean'):
        super().__init__(score_function, model)
        if aggregation_function == 'mean':
            self.aggregation_function = torch.mean
        elif aggregation_function == 'median':
            # self.aggregation_function = torch.median
            self.aggregation_function = lambda x, dim: torch.median(x, dim=dim)[0]
        else:
            self.aggregation_function = aggregation_function

    def train(self, train_dataloader, ensemble_num, subset_num, **kwargs):
        """
        Trains an ensemble of models on randomly sampled subsets of the training data.

        Args:
            train_dataloader (DataLoader): The DataLoader for the training data, 
                                        providing batches of data for training.
            ensemble_num (int): The number of models to ensemble.
            subset_num (int): The size of the subset of data for training each individual model in the ensemble.
            **kwargs: Additional parameters for the :func:`score_predictor.train` method.
                - criterion (callable, optional): Loss function for training. If not provided, uses :func:`QuantileLoss`.
                - alpha (float, optional): Significance level (e.g., 0.1) for quantiles, required if :attr:`criterion` is None.
                - epochs (int, optional): Number of training epochs. Default is :math:`100`.
                - lr (float, optional): Learning rate for optimizer. Default is :math:`0.01`.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training; defaults to :func:`torch.optim.Adam`.
                - verbose (bool, optional): If True, displays training progress. Default is True.

        Raises:
            ValueError: If :attr:`criterion` is not provided and :attr:`alpha` is not specified.
        
        Example::
        
            >>> predictor = Ensemble(model, score_predictor, aggregation_function)
            >>> predictor.train(train_dataloader=train_data_loader, ensemble_num=5, subset_num=500)
            
        .. note::
        
            Procedure:
                1. Creates `ensemble_num` models by sampling subsets from the training data.
                2. For each model in the ensemble:
                - Samples a random subset of indices of size `subset_num` from the dataset.
                - Trains a copy of the base model on this subset.
                - Stores each trained model along with the subset indices used for training.

            Post-training:
                Computes and stores the conformal scores on the training dataset for later use in prediction intervals.
        """
        if ensemble_num <= 0:
            raise ValueError("ensemble_num must be greater than 0")
        
        self.model_list = []
        self.indices_list = []

        dataset = train_dataloader.dataset
        batch_size = train_dataloader.batch_size
        for i in range(ensemble_num):
            indices = torch.randint(0, len(dataset), (subset_num,), device=self._device)
            subset = torch.utils.data.Subset(dataset, indices)
            subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
            model_copy = copy.deepcopy(self._model)
            self.score_function.train(subset_dataloader, model=model_copy, **kwargs)
            self.model_list.append(model_copy)
            self.indices_list.append(indices)

        score_list = []
        dataset = train_dataloader.dataset

        with torch.no_grad():
            for idx in range(len(dataset)):
                x, y_truth = dataset[idx]
                x = x.unsqueeze(0).to(self._device)
                y_truth = torch.tensor([y_truth], dtype=torch.float32).to(self._device)

                model_predict = [model(x)
                                 for i, model in enumerate(self.model_list)
                                 if idx not in self.indices_list[i]
                                 ]
                torch.cuda.empty_cache()

                if model_predict:
                    model_predict_tensor = torch.stack(model_predict)
                    aggregated_predict = self.aggregation_function(model_predict_tensor, dim=0)
                    score_list.append([self.calculate_score(aggregated_predict, y_truth)])

        self.scores = torch.tensor(score_list, dtype=torch.float32).to(self._device)

    def predict(self, alpha, x_batch, y_batch_last=None, aggr_pred_last=None):
        """
        Generates conformal prediction intervals for new data.

        Args:
            alpha (float): Significance level for conformal intervals.
            x_batch (Tensor): Batch of input features.
            y_batch_last (Tensor, optional): Labels from the previous batch for score updates.
            aggr_pred_last (Tensor, optional): Aggregated predictions from the previous batch.

        Returns:
            Tuple: A tuple containing:
                - Prediction intervals for the input batch.
                - Aggregated predictions for the input batch.
        """
        if (y_batch_last is None) != (aggr_pred_last is None):
            raise ValueError("y_batch_last and pred_interval_last must either be provided or be None.")
        if y_batch_last is not None:
            update_scores = self.calculate_score(aggr_pred_last, y_batch_last)
            self.scores = torch.cat([self.scores, update_scores], dim=0) if len(self.scores) > 0 else update_scores
            self.scores = self.scores[len(update_scores):]
            
        self.q_hat = self._calculate_conformal_value(self.scores, alpha)
        x_batch = x_batch.to(self._device)

        with torch.no_grad():
            model_predictions = [model(x_batch) for model in self.model_list]
        torch.cuda.empty_cache()

        predictions_tensor = torch.stack(model_predictions)
        aggregated_predict = self.aggregation_function(predictions_tensor, dim=0)
        
        return self.generate_intervals(aggregated_predict, self.q_hat), aggregated_predict

    def evaluate(self, data_loader, alpha, verbose=True):
        """
        Evaluates the performance of the ensemble model on a test dataset by calculating 
        coverage rates and average sizes of the prediction intervals.

        Args:
            data_loader (DataLoader): The DataLoader providing the test data batches.
            alpha (float): The significance level for conformal prediction, which controls 
                        the width of the prediction intervals (e.g., 0.1 for 90% prediction intervals).
            verbose (bool): If True, prints the coverage rate and average size for each batch. 
                            Default is True.

        Returns:
            dict: A dictionary containing:
                - "Total batches": The number of batches evaluated.
                - "Average coverage rate": The average coverage rate across all batches.
                - "Average prediction interval size": The average size of the prediction intervals.
        
        Example::
        
            >>> eval_results = model.evaluate(test_data_loader, alpha=0.1)
            >>> print(eval_results)
            
        .. note::
            Procedure:
            1. Iterates through each batch in the test data.
            2. For each batch:
                - Generates prediction intervals using the `predict` method.
                - Calculates the coverage rate (percentage of true values within the interval).
                - Calculates the average interval size.
                - (Optional) If `verbose` is True, prints batch-wise coverage rate and average size.
            3. Aggregates and returns the overall average coverage rate and interval size across batches.
        """
        coverage_rates = []
        average_sizes = []

        with torch.no_grad():
            y_batch_last = None
            aggr_pred_last = None
            for index, batch in enumerate(data_loader):
                x_batch, y_batch = batch[0].to(self._device), batch[1].to(self._device)
                prediction_intervals, aggr_pred_last = self.predict(alpha, x_batch, y_batch_last, aggr_pred_last)
                y_batch_last = y_batch
                
                batch_coverage_rate = self._metric('coverage_rate')(prediction_intervals, y_batch)
                batch_average_size = self._metric('average_size')(prediction_intervals)

                if verbose:
                    print(
                        f"Batch: {index + 1}, Coverage rate: {batch_coverage_rate:.4f}, Average size: {batch_average_size:.4f}")

                coverage_rates.append(batch_coverage_rate)
                average_sizes.append(batch_average_size)

        avg_coverage_rate = sum(coverage_rates) / len(coverage_rates)
        avg_average_size = sum(average_sizes) / len(average_sizes)

        res_dict = {"Total batches": index + 1,
                    "Coverage_rate": avg_coverage_rate,
                    "Average_size": avg_average_size}

        return res_dict
