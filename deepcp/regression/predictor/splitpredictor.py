



from tqdm import  tqdm
import torch

class SplitPredictor(object):
    def __init__(self, score_function, model, device):
        self.score_function = score_function
        self._model = model
        self._device = device

    def calibrate(self, cal_dataloader, alpha):
        predicts_labels = [
            (self._logits_transformation(self._model(examples[0])).detach().cpu(),
             examples[1])
            for examples in tqdm(cal_dataloader)
        ]
        predicts, labels = map(
            lambda x: torch.stack(x).float(),
            zip(*predicts_labels)
        )
        self.calculate_threshold(predicts, labels, alpha)

    def calculate_threshold(self, predicts, labels, alpha):
        self.scores = self.score_function(predicts, labels)


    def predict(self, x):
        perdicts = self._model(x)

