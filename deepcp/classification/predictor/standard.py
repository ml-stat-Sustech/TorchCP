
import torch

from .base import BasePredictor



class StandardPredictor(BasePredictor):
    def __init__(self, score_function):
        super().__init__(score_function)
        
    
    def fit(self, x_cal, y_cal, alpha):
        scores= []
        for index,(x,y) in enumerate(zip(x_cal,y_cal)):
            scores.append(self.score_function(x,y))
        scores = torch.tensor(scores)
        self.q_hat = torch.quantile(scores,1-alpha)


        
    def predict(self, x):
        scores = self.score_function.predict(x)
        S = torch.argwhere(scores < self.q_hat).reshape(-1).tolist()
        return S