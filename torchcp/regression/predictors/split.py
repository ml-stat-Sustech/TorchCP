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
    Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
    paper: https://arxiv.org/abs/1604.04173
    
    :param model: a pytorch model for regression.
    """

    def __init__(self, model =None):
        self._model = model
        if self._model != None:
            assert isinstance(model, nn.Module), "The model is not an instance of torch.nn.Module"
            self._device = get_device(model)
        else:
            self._device = None
        self._metric = Metrics()

    def _train(self, model, epochs, train_dataloader, criterion, optimizer, verbose=True):
        model.train()
        device = get_device(model)
        if verbose:
            with tqdm(total=epochs, desc = "Epoch") as _tqdm:
                for epoch in range(epochs):
                    running_loss = 0.0
                    for index, (tmp_x, tmp_y) in enumerate(train_dataloader):
                        outputs = model(tmp_x.to(device))
                        loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss = (running_loss*max(0,index) + loss.data.cpu().numpy())/(index+1)
                        _tqdm.set_postfix({  "loss": '{:.6f}'.format(running_loss)})
                    _tqdm.update(1)
                
        else:
            for index, (tmp_x, tmp_y) in enumerate(train_dataloader):
                outputs = model(tmp_x.to(device))
                loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 


                 
        # else:
            

        #     if verbose:
        #         print(f"Epoch {epoch + 1} completed, Average Loss: {running_loss / len(train_dataloader):.6f}")

        print("Finish training!")
        model.eval()

    def fit(self, train_dataloader, **kwargs):
        model = kwargs.get('model', self._model)
        epochs = kwargs.get('epochs', 100)
        criterion = kwargs.get('criterion', nn.MSELoss())
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._train(model, epochs, train_dataloader, criterion, optimizer, verbose)

    def calculate_score(self, predicts, y_truth):
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.abs(predicts - y_truth)

    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        predicts_list = []
        y_truth_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)
            predicts = torch.cat(predicts_list).float().to(self._device)
            y_truth = torch.cat(y_truth_list).to(self._device)
        self.calculate_threshold(predicts, y_truth, alpha)

    def calculate_threshold(self, predicts, y_truth, alpha):
        scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    def predict(self, x_batch):
        self._model.eval()
        x_batch.to(self._device)
        with torch.no_grad():
            predicts_batch = self._model(x_batch)
            return self.generate_intervals(predicts_batch, self.q_hat)

    def generate_intervals(self, predicts_batch, q_hat):
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))

        prediction_intervals[..., 0] = predicts_batch - q_hat.view(1, q_hat.shape[0])
        prediction_intervals[..., 1] = predicts_batch + q_hat.view(1, q_hat.shape[0])

        return prediction_intervals

    def evaluate(self, data_loader):
        y_list = []
        predict_list = []
        with torch.no_grad():
            for examples in data_loader:
                tmp_x, tmp_y = examples[0].to(self._device), examples[1].to(self._device)
                tmp_prediction_intervals = self.predict(tmp_x)
                y_list.append(tmp_y)
                predict_list.append(tmp_prediction_intervals)

        predicts = torch.cat(predict_list, dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)

        res_dict = {"Coverage_rate": self._metric('coverage_rate')(predicts, test_y),
                    "Average_size": self._metric('average_size')(predicts)}
        return res_dict
