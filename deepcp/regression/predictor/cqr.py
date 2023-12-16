



from tqdm import  tqdm
import torch
import numpy as np


from deepcp.regression.utils.metrics import Metrics
from deepcp.regression.predictor.splitpredictor import SplitPredictor


class CQR(SplitPredictor):
    def __init__(self, model, device):
        self._model = model
        self._device = device
        self._metric = Metrics()


    def calculate_threshold(self, predicts, y_truth, alpha):
        self.scores = torch.maximum(predicts[:,0]-y_truth, y_truth - predicts[:,1])
        self.q_hat = torch.quantile(self.scores,  (1 - alpha) *(1+1/self.scores.shape[0]))
        
        
    def predict(self, x_batch):
        predicts_batch = self._model(x_batch.to(self._device)).float()
        lower_bound = predicts_batch[:,0] - self.q_hat
        upper_bound = predicts_batch[:,1] + self.q_hat
        prediction_intervals =  torch.stack([lower_bound, upper_bound],dim=1)
        return prediction_intervals

        
    

