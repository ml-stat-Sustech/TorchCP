



from tqdm import  tqdm
import torch
import numpy as np


from deepcp.regression.utils.metrics import Metrics
from deepcp.regression.predictor.splitpredictor import SplitPredictor


class ACI(SplitPredictor):
    """_summary_

    :param model: a deep learning model that can output alpha/2 and 1-alpha/2 quantile regression.
    """
    def __init__(self, model, gamma):
        self._model = model
        self._device = self._model.device
        self._metric = Metrics()
        self.__gamma = gamma


    def calculate_threshold(self, predicts, y_truth, alpha):
        self.scores = torch.maximum(predicts[:,0]-y_truth, y_truth - predicts[:,1])
        self.q_hat = torch.quantile(self.scores,  (1 - alpha) *(1+1/self.scores.shape[0]))
        
        # the alpha of the previous time
        self.alpha_t = alpha 
        # Desired significance level
        self.alpha = alpha 
        
        
    def predict(self, x, y_t, pred_interval_t):
        """
        
        : param y_t: the truth value at the time t.
        : param pred_interval_t: the prediction interval for the time t.
        
        """
        err_t = 1 if (y_t >= pred_interval_t[0]) & (y_t <= pred_interval_t[1]) else 0
        self.alpha_t = self.alpha_t +  self.__gamma(self.alpha - err_t)
        predicts_batch = self._model(x.to(self._device)).float()
        q_hat = torch.quantile(self.scores,  (1 - self.alpha_t) *(1+1/self.scores.shape[0]))
        lower_bound = predicts_batch[:,0] - q_hat
        upper_bound = predicts_batch[:,1] + q_hat
        prediction_intervals =  torch.stack([lower_bound, upper_bound],dim=1)
        return prediction_intervals

        
    

