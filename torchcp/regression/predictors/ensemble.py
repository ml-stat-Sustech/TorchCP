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


class Ensemble(object):
    """
    Ensemble Conformal Prediction Interval

    There are two main implementations for `score_predictor`:
    
    1. EnbPI:
        - Paper: Conformal Prediction Interval for Dynamic Time-Series (Xu and Xie, 2020)
        - Link: https://proceedings.mlr.press/v139/xu21h/xu21h.pdf
        - Github: https://github.com/hamrel-cxu/EnbPI
    
    2. EnCQR:
        - Paper: Ensemble Conformalized Quantile Regression for Probabilistic 
                Time Series Forecasting (Jensen et al., 2022)
        - Link: https://ieeexplore.ieee.org/abstract/document/9940232
        - Github: https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression
       
    Args:
        model (torch.nn.Module): The base model to be used in the ensemble.
        score_predictor (torchcp.regression.predictors): The method for calculating scores and prediction intervals.
                        Can be :class:`torchcp.regression.predictors.SplitPredictor` for EnbPI, 
                        :class:`torchcp.regression.predictors.CQR` for EnCQR, or other variants.
        aggregation_function (str or callable): A function to aggregate predictions from multiple 
                        models in the ensemble. Options include:
                        - torch.mean: Computes the mean of the predictions.
                        - torch.median: Computes the median of the predictions.
                        - Custom function: Should accept a tensor and dimension as input, returning the result.

    Example::
    
        >>> ########################## EnbPI ###########################
        >>> model = build_regression_model("NonLinearNet")(X.shape[1], 1, 64, 0.5).to(device)
        >>> score_predictor = SplitPredictor(model=None)
        >>> aggregation_function = torch.mean
        >>> ensemble = Ensemble(model, score_predictor, aggregation_function=aggregation_function)
        
        >>> ########################## EnCQR  ###########################
        >>> model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
        >>> score_predictor = CQR(model=None)
        >>> aggregation_function = torch.mean
        >>> ensemble = Ensemble(model, score_predictor, aggregation_function=aggregation_function)
    """
    def __init__(self, model, score_predictor, aggregation_function='mean'):
        self._model = model
        self._device = get_device(model)
        self._metric = Metrics()
        self.score_predictor = score_predictor
        
        if aggregation_function == 'mean':
            self.aggregation_function = torch.mean
        elif aggregation_function == 'median':
            self.aggregation_function = torch.median
        else:
            self.aggregation_function = aggregation_function

    def fit(self, train_dataloader, ensemble_num, subset_num, **kwargs):
        """
        Trains an ensemble of models on randomly sampled subsets of the training data.

        Args:
            train_dataloader (DataLoader): The DataLoader for the training data, 
                                        providing batches of data for training.
            ensemble_num (int): The number of models to ensemble.
            subset_num (int): The size of the subset of data for training each individual model in the ensemble.
            **kwargs: Additional parameters for the :func:`score_predictor.fit` method.
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
            >>> predictor.fit(train_dataloader=train_data_loader, ensemble_num=5, subset_num=500)
            
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
        self.model_list = []
        self.indices_list = []

        dataset = train_dataloader.dataset
        batch_size = train_dataloader.batch_size
        for i in range(ensemble_num):
            indices = torch.randint(0, len(dataset), (subset_num,), device=self._device)
            subset = torch.utils.data.Subset(dataset, indices)
            subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
            model_copy = copy.deepcopy(self._model)
            self.score_predictor.fit(subset_dataloader, model=model_copy, **kwargs)
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
                    score_list.append([self.score_predictor.calculate_score(aggregated_predict, y_truth)])

        self.scores = torch.tensor(score_list, dtype=torch.float32).to(self._device)

    def predict(self, alpha, x_batch, y_batch=None):
        """
        Generates prediction intervals for a given batch of data using the conformal prediction framework.
        
        Args:
            alpha (float): The significance level for conformal prediction, controlling the size of the prediction intervals.
            x_batch (torch.Tensor): The input batch of data for which predictions are to be generated.
            y_batch (torch.Tensor, optional): The true labels corresponding to the input batch. 
                    If provided, scores will be updated based on the true labels. Default is None.

        Returns:
            intervals (torch.Tensor): The generated prediction intervals for the input data based on the conformal prediction framework.
        """
        self.q_hat = self.score_predictor._calculate_conformal_value(self.scores, alpha)
        x_batch = x_batch.to(self._device)

        with torch.no_grad():
            model_predictions = [model(x_batch) for model in self.model_list]
        torch.cuda.empty_cache()

        predictions_tensor = torch.stack(model_predictions)
        aggregated_predict = self.aggregation_function(predictions_tensor, dim=0)
        
        if y_batch is not None:
            update_scores = self.score_predictor.calculate_score(aggregated_predict, y_batch)
            self.scores = torch.cat([self.scores, update_scores], dim=0) if len(self.scores) > 0 else update_scores
            self.scores = self.scores[len(update_scores):]
        
        return self.score_predictor.generate_intervals(aggregated_predict, self.q_hat)

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
            for index, batch in enumerate(data_loader):
                x_batch, y_batch = batch[0].to(self._device), batch[1].to(self._device)
                prediction_intervals = self.predict(alpha, x_batch, y_batch)

                batch_coverage_rate = self._metric('coverage_rate')(prediction_intervals, y_batch)
                batch_average_size = self._metric('average_size')(prediction_intervals)

                if verbose:
                    print(
                        f"Batch: {index + 1}, Coverage rate: {batch_coverage_rate.item():.4f}, Average size: {batch_average_size.item():.4f}")

                coverage_rates.append(batch_coverage_rate)
                average_sizes.append(batch_average_size)

        avg_coverage_rate = sum(coverage_rates) / len(coverage_rates)
        avg_average_size = sum(average_sizes) / len(average_sizes)

        res_dict = {"Total batches": index + 1,
                    "Average coverage rate": avg_coverage_rate.item(),
                    "Average prediction interval size": avg_average_size.item()}

        return res_dict
