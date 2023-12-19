#
# Reference paper: "Conformal Prediction Under Covariate Shift"
#

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from deepcp.classification.predictor.induction import InductivePredictor
from deepcp.classification.predictor.utils import build_DomainDetecor, IW


class WeightedPredictor(InductivePredictor):
    """
    Weighted conformal predictor (Tibshirani et al., 2019)
    paper : https://arxiv.org/abs/1904.06019
    """

    def __init__(self, score_function, model, image_encoder):
        super().__init__(score_function, model)

        self.image_encoder = image_encoder
        #  non-conformity scores
        self.scores = None
        # important weight
        self.IWeight = None
        # significance level
        self.alpha = None

    def calibrate(self, cal_dataloader, alpha):
        self.alpha = alpha
        self.cal_dataloader = cal_dataloader
        logits_list = []
        labels_list = []
        cal_features_list = []
        with torch.no_grad():
            for examples in tqdm(cal_dataloader):
                tmp_x, tmp_labels = examples[0].to(self._model_device), examples[1]
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                cal_features_list.append(self.image_encoder(tmp_x))
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
            cal_features = torch.cat(cal_features_list).float()

        self.source_image_features = cal_features

        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        self.scores = torch.zeros(logits.shape[0] + 1)
        for index, (x, y) in enumerate(zip(logits, labels)):
            self.scores[index] = self.score_function(x, y)
        self.scores[index + 1] = torch.tensor(torch.inf)
        self.scores_sorted = torch.tensor(self.scores).to(self._model_device).sort()[0]

    def predict(self, x_batch):
        bs = x_batch.shape[0]
        with torch.no_grad():
            image_features = self.image_encoder(x_batch.to(self._model_device)).float()
            print(image_features.shape)
            w_new = self.IW(image_features)

            w_sorted = self.w_sorted.expand([bs, -1])
            w_sorted = torch.cat([w_sorted, w_new.unsqueeze(1)], 1)
            p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
            p_sorted_acc = p_sorted.cumsum(1)

            i_T = torch.argmax((p_sorted_acc >= 1.0 - self.alpha).int(), dim=1, keepdim=True)
            q_hat_batch = self.scores_sorted.expand([bs, -1]).gather(1, i_T).detach()

        logits = self._model(x_batch.to(self._model_device)).float()
        logits_batch = self._logits_transformation(logits).detach()
        sets = []
        for index, (logits, q_hat) in enumerate(zip(logits_batch, q_hat_batch)):
            sets.append(self.predict_with_probs(logits, q_hat))
        return sets

    def evaluate(self, val_dataloader):
        ###############
        # train domain classifier
        ###############
        print(f'Training a domain classifier')
        val_features_list = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_labels = examples[0].to(self._model_device), examples[1]
                val_features_list.append(self.image_encoder(tmp_x))
            target_image_features = torch.cat(val_features_list).float()

        ###############################
        # Train domain detector
        ###############################
        source_labels = torch.zeros(self.source_image_features.shape[0])
        target_labels = torch.ones(target_image_features.shape[0])

        input = torch.cat((self.source_image_features, target_image_features))
        labels = torch.cat((source_labels, target_labels))
        dataset = torch.utils.data.TensorDataset(input.float(), labels.float().long())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=False)

        domain_detecor = build_DomainDetecor(target_image_features.shape[1], 2, self._model_device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(domain_detecor.parameters(), lr=0.001)

        epochs = 10
        for epoch in range(epochs):
            loss_log = 0
            accuracy_log = 0
            for X_train, y_train in data_loader:
                y_train = y_train.to(self._model_device)
                outputs = domain_detecor(X_train.to(self._model_device))
                loss = criterion(outputs, y_train.view(-1))
                loss_log += loss.item() / len(data_loader)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum(predictions == y_train.view(-1)).item() / len(y_train)
                accuracy_log += accuracy / len(data_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.IW = IW(domain_detecor, self._model_device)
        w_cal = self.IW(self.source_image_features.to(self._model_device))
        self.w_sorted = w_cal.sort(descending=False)[0]

        prediction_sets = []
        labels_list = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_label = examples[0], examples[1]
                prediction_sets_batch = self.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)

        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_sets, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_sets, val_labels)
        return res_dict
