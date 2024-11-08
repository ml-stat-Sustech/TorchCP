import torch
import torch.optim as optim

from .split import SplitPredictor
from ..loss import QuantileLoss


class CQR(SplitPredictor):
    """
    Conformalized Quantile Regression (CQR) for creating prediction intervals with specified 
    coverage levels using quantile regression.

    Args:
        model (torch.nn.Module): A pytorch regression model that can output alpha/2 and 1-alpha/2 quantiles.

    Reference:
        Paper: Conformalized Quantile Regression (Romano et al., 2019)
        Link: https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf
        Github: https://github.com/yromano/cqr
    """

    def __init__(self, model):
        super().__init__(model)

    def fit(self, train_dataloader, **kwargs):
        """
        Trains the model on provided training data with :math:`[alpha/2, 1-alpha/2]` quantile regression loss.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
            **kwargs: Additional training parameters.
                - model (torch.nn.Module, optional): Model to be trained; defaults to the model passed to the predictor.
                - criterion (callable, optional): Loss function for training. If not provided, uses :func:`QuantileLoss`.
                - alpha (float, optional): Significance level (e.g., 0.1) for quantiles, required if :attr:`criterion` is None.
                - epochs (int, optional): Number of training epochs. Default is :math:`100`.
                - lr (float, optional): Learning rate for optimizer. Default is :math:`0.01`.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training; defaults to :func:`torch.optim.Adam`.
                - verbose (bool, optional): If True, displays training progress. Default is True.

        Raises:
            ValueError: If :attr:`criterion` is not provided and :attr:`alpha` is not specified.
            
        .. note::
            This function is optional but recommended, because the training process for each preditor's model is different. 
            We provide a default training method, and users can change the hyperparameters :attr:`kwargs` to modify the training process.
            If the fit function is not used, users should pass the trained model to the predictor at the beginning.
        """
        model = kwargs.get('model', self._model)
        criterion = kwargs.get('criterion', None)

        if criterion is None:
            alpha = kwargs.get('alpha', None)
            if alpha is None:
                raise ValueError("When 'criterion' is not provided, 'alpha' must be specified.")
            quantiles = [alpha / 2, 1 - alpha / 2]
            criterion = QuantileLoss(quantiles)

        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._train(model, epochs, train_dataloader, criterion, optimizer, verbose)

    def calculate_score(self, predicts, y_truth):
        if len(predicts.shape) == 2:
            predicts = predicts.unsqueeze(1)
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.maximum(predicts[..., 0] - y_truth, y_truth - predicts[..., 1])

    def predict(self, x_batch):
        self._model.eval()
        if len(x_batch.shape) == 1:
            x_batch = x_batch.unsqueeze(0)
        predicts_batch = self._model(x_batch.to(self._device)).float()

        return self.generate_intervals(predicts_batch, self.q_hat)

    def generate_intervals(self, predicts_batch, q_hat):
        """
        Constructs the prediction intervals based on model predictions and the conformal threshold.

        Args:
            predicts_batch (torch.Tensor): Predicted quantile intervals from the model, shape (batch_size, 2).
            q_hat (torch.Tensor): Conformal threshold calculated from the calibration data, shape (1,).

        Returns:
            torch.Tensor: Adjusted prediction intervals, shape (batch_size, 2).
        """
        if len(predicts_batch.shape) == 2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch[..., 0] - q_hat.view(1, q_hat.shape[0], 1)
        prediction_intervals[..., 1] = predicts_batch[..., 1] + q_hat.view(1, q_hat.shape[0], 1)
        return prediction_intervals
