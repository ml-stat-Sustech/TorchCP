



from tqdm import  tqdm
import torch
import numpy as np


from deepcp.regression.utils.metrics import Metrics


class SplitPredictor(object):
    def __init__(self, score_function, model, device):
        self.score_function = score_function
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

    def calculate_threshold(self, predicts, labels, alpha):
        self.scores = self.score_function(predicts, labels)
        
        self.q_hat = torch.quantile(self.scores, np.ceil((self.scores.shape[0]/2+ 1) * (1 - alpha)) / self.scores.shape[0])
        print(self.q_hat)
        
        
    def predict(self, x_batch):
        predicts_batch = self._model(x_batch.to(self._device)).float()
        predicts = torch.zeros(len(x_batch))
        for index, predict in enumerate(predicts_batch):
            predicts[index] = predict
            
        lower_bound = predicts - self.q_hat
        upper_bound = predicts + self.q_hat
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

        
    

