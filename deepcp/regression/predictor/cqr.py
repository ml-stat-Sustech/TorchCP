



from tqdm import  tqdm
import torch
import numpy as np


from deepcp.regression.utils.metrics import Metrics
from deepcp.regression.predictor.splitpredictor import SplitPredictor


class CQR(object):
    def __init__(self, model, device):
        self._model = model
        self._device = device
        self._metric = Metrics()

    def calibrate(self, cal_dataloader, alpha):
        predicts_list = []
        labels_list = []
        x_list = []
        with torch.no_grad():
            for  examples in tqdm(cal_dataloader):
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1]
                tmp_predicts = self._model(tmp_x).detach().cpu()
                x_list.append(tmp_x)
                predicts_list.append(tmp_predicts)
                labels_list.append(tmp_labels)
            predicts = torch.cat(predicts_list).float()
            labels = torch.cat(labels_list)
            x = torch.cat(x_list).float()
        self.predicts = predicts
        self.labels = labels
        self.x = x
        self.calculate_threshold(predicts, labels, alpha)

    def calculate_threshold(self, predicts, y_truth, alpha):
        self.scores = torch.maximum(predicts[:,0]-y_truth, y_truth - predicts[:,1])
        self.q_hat = torch.quantile(self.scores,  (1 - alpha) *(1+1/self.scores.shape[0]))
        
        
    def predict(self, x_batch):
        predicts_batch = self._model(x_batch.to(self._device)).float()
        lower_bound = predicts_batch[:,0] - self.q_hat
        upper_bound = predicts_batch[:,1] + self.q_hat
        prediction_intervals =  torch.stack([lower_bound, upper_bound],dim=1)
        return prediction_intervals

    def evaluate(self, data_loader):
        predicts_list = []
        labels_list = []
        with torch.no_grad():
            for  examples in tqdm(data_loader):
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1]
                tmp_logits = self._model(tmp_x).detach().cpu()
                predicts_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            predicts = torch.cat(predicts_list).float()
            val_labels = torch.cat(labels_list)
        lower_bound = predicts - self.q_hat
        upper_bound = predicts + self.q_hat
        prediction_intervals =  torch.hstack([lower_bound, upper_bound])
    
        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_intervals, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_intervals, val_labels)
        return res_dict

        
    

