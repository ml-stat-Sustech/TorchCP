#
# Reference paper: "Conformal Prediction Under Covariate Shift"
#

import torch
import torch.nn as nn
import torch.optim as optim

from .split import SplitPredictor
from .utils import build_DomainDetecor, IW


class WeightedPredictor(SplitPredictor):
    """
    Method: Weighted Conformal Prediction
    Paper: Conformal Prediction Under Covariate Shift (Tibshirani et al., 2019)
    Link: https://arxiv.org/abs/1904.06019
    Github: https://github.com/ryantibs/conformal/
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module): A PyTorch model.
        image_encoder (torch.nn.Module): A PyTorch model to generate the embedding feature of an input image.
        domain_classifier (torch.nn.Module, optional): A PyTorch model (a binary classifier) to predict the probability that an embedding feature comes from the source domain. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
    """

    def __init__(self, score_function, model=None, temperature=1, image_encoder=None, domain_classifier=None):
        
        super().__init__(score_function, model, temperature)

        if image_encoder is None:
            raise ValueError("image_encoder cannot be None.")
        
        self.image_encoder = image_encoder.to(self._device)
        self.domain_classifier = domain_classifier

        #  non-conformity scores
        self.scores = None
        # significance level
        self.alpha = None
        # Domain Classifier

    def calibrate(self, cal_dataloader, alpha):
        """
        Calibrate the model using the calibration set.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
            alpha (float): The significance level.
        """
        logits_list = []
        labels_list = []
        cal_features_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                cal_features_list.append(self.image_encoder(tmp_x))
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
            self.source_image_features = torch.cat(cal_features_list).float()

        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        """
        Calculate the conformal prediction threshold.

        Args:
            logits (torch.Tensor): The logits output from the model.
            labels (torch.Tensor): The ground truth labels.
            alpha (float): The significance level.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")
        
        self.alpha = alpha
        self.scores = torch.zeros(logits.shape[0] + 1).to(self._device)
        self.scores[:logits.shape[0]] = self.score_function(logits, labels)
        self.scores[logits.shape[0]] = torch.tensor(torch.inf).to(self._device)
        self.scores_sorted = self.scores.sort()[0]

    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """
        
        
        bs = x_batch.shape[0]
        with torch.no_grad():
            image_features = self.image_encoder(x_batch.to(self._device)).float()
            if self.domain_classifier == None:
                self._train_domain_classifier(image_features)

            self.IW = IW(self.domain_classifier).to(self._device)
            w_new = self.IW(image_features)

            w_sorted = self.w_sorted.expand([bs, -1])
            w_sorted = torch.cat([w_sorted, w_new.unsqueeze(1)], 1)
            p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
            p_sorted_acc = p_sorted.cumsum(1)

            i_T = torch.argmax((p_sorted_acc >= 1.0 - self.alpha).int(), dim=1, keepdim=True)
            q_hat_batch = self.scores_sorted.expand([bs, -1]).gather(1, i_T).detach()

        logits = self._model(x_batch.to(self._device)).float()
        logits = self._logits_transformation(logits).detach()
        sets = []
        for index, (logits_instance, q_hat) in enumerate(zip(logits, q_hat_batch)):
            sets.extend(self.predict_with_logits(logits_instance, q_hat))
        return sets

    def evaluate(self, val_dataloader):
        """
        Evaluate the prediction sets on a validation dataset.

        This function trains a domain classifier if it is not already provided, and then uses it to compute importance weights for the validation set. It then generates prediction sets for the validation set and computes evaluation metrics.

        Args:
            val_dataloader (torch.utils.data.DataLoader): A dataloader of the validation set.

        Returns:
            dict: A dictionary containing the coverage rate and average size of the prediction sets.
        """
        
        ###############
        # train domain classifier
        ###############
        print(f'Training a domain classifier')
        val_features_list = []
        with torch.no_grad():
            for examples in val_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                val_features_list.append(self.image_encoder(tmp_x))
            target_image_features = torch.cat(val_features_list).float()

        ###############################
        # Training domain detector
        ###############################
        if self.domain_classifier == None:
            self._train_domain_classifier(target_image_features)

        self.IW = IW(self.domain_classifier).to(self._device)
        w_cal = self.IW(self.source_image_features.to(self._device))
        self.w_sorted = w_cal.sort(descending=False)[0]

        prediction_sets = []
        labels_list = []
        with torch.no_grad():
            for examples in val_dataloader:
                tmp_x, tmp_label = examples[0].to(self._device), examples[1].to(self._device)
                prediction_sets_batch = self.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)

        result_dict = {"Coverage_rate": self._metric('coverage_rate')(prediction_sets, val_labels),
                       "Average_size": self._metric('average_size')(prediction_sets, val_labels)}
        return result_dict
    
    def _train_domain_classifier(self, target_image_features):
        source_labels = torch.zeros(self.source_image_features.shape[0]).to(self._device)
        target_labels = torch.ones(target_image_features.shape[0]).to(self._device)

        input = torch.cat((self.source_image_features, target_image_features))
        labels = torch.cat((source_labels, target_labels))
        dataset = torch.utils.data.TensorDataset(input.float(), labels.float().long())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=False)

        self.domain_classifier = build_DomainDetecor(target_image_features.shape[1], 2, self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.001)

        epochs = 5
        for epoch in range(epochs):
            loss_log = 0
            accuracy_log = 0
            for X_train, y_train in data_loader:
                y_train = y_train.to(self._device)
                outputs = self.domain_classifier(X_train.to(self._device))
                loss = criterion(outputs, y_train.view(-1))
                loss_log += loss.item() / len(data_loader)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum((predictions == y_train.view(-1))).item() / len(y_train)
                accuracy_log += accuracy / len(data_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()