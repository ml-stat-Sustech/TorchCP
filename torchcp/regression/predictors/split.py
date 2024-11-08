# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from torchcp.utils.common import calculate_conformal_value
from torchcp.utils.common import get_device
from ..utils.metrics import Metrics


class SplitPredictor(object):
    """
    Split Conformal Prediction for Regression.
    
    This predictor allows for the construction of a prediction band for the response 
    variable using any estimator of the regression function.
        
    Args:
        model (torch.nn.Module): A pytorch regression model that can output predicted point.
        
    Reference:
        Paper: Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
        Link: https://arxiv.org/abs/1604.04173
        Github: https://github.com/ryantibs/conformal
    """

    def __init__(self, model=None):
        self._model = model
        if self._model is not None:
            assert isinstance(model, nn.Module), "The model must be an instance of torch.nn.Module"
            self._device = get_device(model)
        else:
            self._device = None
        self._metric = Metrics()

    def _train(self, model, epochs, train_dataloader, criterion, optimizer, verbose=True):
        """
        Trains the given model using the provided training data loader, criterion, and optimizer.
        
        Args:
            model (torch.nn.Module): The model to be trained.
            epochs (int): The number of epochs to train the model.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
            verbose (bool, optional): If True, displays a progress bar and loss information. Defaults to True.
        """
        
        model.train()
        device = get_device(model)
        if verbose:
            with tqdm(total=epochs, desc="Epoch") as _tqdm:
                for epoch in range(epochs):
                    running_loss = 0.0
                    for index, (tmp_x, tmp_y) in enumerate(train_dataloader):
                        outputs = model(tmp_x.to(device))
                        loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss = (running_loss * max(0, index) + loss.data.cpu().numpy()) / (index + 1)
                        _tqdm.set_postfix({"loss": f"{running_loss:.6f}"})
                    _tqdm.update(1)
        else:
            for tmp_x, tmp_y in train_dataloader:
                outputs = model(tmp_x.to(device))
                loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
        print("Training complete.")
        model.eval()

    def fit(self, train_dataloader, **kwargs):
        """
        Trains the model using the provided training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): The model to be trained. Defaults to the model passed to the predictor.
                - epochs (int, optional): Number of training epochs. Defaults to :math:`100`.
                - criterion (nn.Module, optional): Loss function. Defaults to :func:`torch.nn.MSELoss()`.
                - lr (float, optional): Learning rate for the optimizer. Defaults to :math:`0.01`.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to :func:`torch.optim.Adam`.
                - verbose (bool, optional): If True, prints training progress. Defaults to True.
            
        .. note::
            This function is optional but recommended, because the training process for each preditor's model is different. 
            We provide a default training method, and users can change the hyperparameters :attr:`kwargs` to modify the training process.
            If the fit function is not used, users should pass the trained model to the predictor at the beginning.
        """
        
        model = kwargs.get('model', self._model)
        epochs = kwargs.get('epochs', 100)
        criterion = kwargs.get('criterion', nn.MSELoss())
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._train(model, epochs, train_dataloader, criterion, optimizer, verbose)

    def calculate_score(self, predicts, y_truth):
        """
        Calculates the score used for conformal prediction, which measures the deviation 
        of the true values from the predicted intervals.

        Args:
            predicts (torch.Tensor): Tensor of predicted quantile intervals, shape (batch_size, 2).
            y_truth (torch.Tensor): Tensor of true target values, shape (batch_size,).

        Returns:
            torch.Tensor: Tensor of non-conformity scores indicating the deviation of predictions from true values.
        """
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.abs(predicts - y_truth)

    def calibrate(self, cal_dataloader, alpha):
        """
        Calibrate the predictor using a calibration dataset and a given significance level :attr:`alpha`.
        
        Args:
            cal_dataloader (torch.utils.data.DataLoader): The dataloader containing the calibration dataset.
            alpha (float): The significance level for calibration. Should be in the range (0, 1).
        """
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
        self.calculate_threshold(predicts, y_truth, alpha)

    def calculate_threshold(self, predicts, y_truth, alpha):
        """
        Calculates the threshold for conformal prediction.

        Args:
            predicts (torch.Tensor): Predicted bin probabilities, shape (batch_size, num_bins).
            y_truth (torch.Tensor): Ground truth values, shape (batch_size,).
            alpha (float): Desired significance level.

        Sets:
            self.q_hat: Calculated threshold for prediction intervals.
        """
        scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    def predict(self, x_batch):
        """
        Generates prediction intervals for a batch of input data.

        Args:
            x_batch (torch.Tensor): Input batch of data points, shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Prediction intervals, shape (batch_size, 2).
        """
        self._model.eval()
        x_batch.to(self._device)
        with torch.no_grad():
            predicts_batch = self._model(x_batch)
            return self.generate_intervals(predicts_batch, self.q_hat)

    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generate prediction intervals by adjusting predictions with the calibrated :attr:`q_hat` threshold.
        
        Args:
            predicts_batch (torch.Tensor): A batch of predictions with shape (batch_size, ...).
            q_hat (torch.Tensor): A tensor containing the calibrated thresholds with shape (num_thresholds,).
        Returns:
            torch.Tensor: A tensor containing the prediction intervals with shape (batch_size, num_thresholds, 2).
                          The last dimension represents the lower and upper bounds of the intervals.
        """
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch - q_hat.view(1, q_hat.shape[0])
        prediction_intervals[..., 1] = predicts_batch + q_hat.view(1, q_hat.shape[0])
        return prediction_intervals

    def evaluate(self, data_loader):
        """
        Evaluate the model on a test dataloader, returning coverage rate and interval size.
        
        Args:
            data_loader (torch.utils.data.DataLoader): The dataloader containing the test dataset.
            
        Returns:
            dict: A dictionary containing the coverage rate and average interval size with keys:
            - Coverage_rate (float): The coverage rate of the prediction intervals.
            - Average_size (float): The average size of the prediction intervals.
        """
        y_list, predict_list = [], []
        with torch.no_grad():
            for tmp_x, tmp_y in data_loader:
                tmp_x, tmp_y = tmp_x.to(self._device), tmp_y.to(self._device)
                tmp_prediction_intervals = self.predict(tmp_x)
                y_list.append(tmp_y)
                predict_list.append(tmp_prediction_intervals)

        predicts = torch.cat(predict_list, dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)

        res_dict = {
            "Coverage_rate": self._metric('coverage_rate')(predicts, test_y),
            "Average_size": self._metric('average_size')(predicts)
        }
        return res_dict
