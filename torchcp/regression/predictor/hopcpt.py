# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from typing import (
    Sequence,
    TypeVar
)
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset

from torchcp.regression.predictor.split import SplitPredictor
from torchcp.regression.utils.model import EncodingNetwork, HopfieldAssociation
from torchcp.regression.loss import HopCPTLoss

class HopCPTPredictor(SplitPredictor):
    def __init__(self, score_function, model=None, encoding_model=None, Hopfield_model=None):
        super().__init__(score_function, model)
        self.encoding_model = encoding_model
        self.Hopfield_model = Hopfield_model
        
    def calibrate(self, cal_dataloader, alpha, split_ratio=0.6):
        train_hopcpt_loader, cal_hopcpt_loader = self._split_dataloader(cal_dataloader, split_ratio)
        if self.encoding_model == None or self.Hopfield_model == None:
            self.train_hopcpt(train_hopcpt_loader)
        
        self._model.eval()
        x_list, predicts_list, y_truth_list = [], [], []
        with torch.no_grad():
            for tmp_x, tmp_labels in cal_hopcpt_loader:
                tmp_x, tmp_labels = tmp_x.to(self._device), tmp_labels.to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)
                x_list.append(tmp_x)

        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)
        cal_x = torch.cat(x_list).to(self._device)
        self.scores = self.calculate_score(predicts, y_truth)
        
        self.alpha = alpha
        self.cal_x = cal_x
        
    def predict(self, x_batch, sampling_num=200):
        attention_weight = self._compute_attention_weights(self.cal_x, x_batch)
        q_hat = self._calculate_conformal_value(attention_weight, self.scores, self.alpha, sampling_num)
        
        self._model.eval()
        x_batch.to(self._device)
        with torch.no_grad():
            predicts_batch = self._model(x_batch)
        return self.generate_intervals(predicts_batch, q_hat)
        
    def _calculate_conformal_value(self, attention_weight, scores, alpha, sampling_num):
        """
        Calculate the conformal values based on attention weights and scores.
        
        Args:
            attention_weight (torch.Tensor): A tensor of shape [batch_size, t] representing the attention weights for each sample.
            scores (torch.Tensor): A tensor of shape [t,] representing the scores.
            alpha (float): The significance level for calculating the quantiles.
            sampling_num (int): The number of samples to draw from the attention-weighted distribution.
        
        Returns:
            torch.Tensor: A tensor of shape [batch_size, 2] containing the quantiles (q_hat) for each sample.
        """
        sampling_score = torch.multinomial(attention_weight, sampling_num, replacement=True)
        sampled_scores = scores[sampling_score]
        
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        q_hat = torch.stack([
            sampled_scores.quantile(lower_quantile, dim=1), 
            sampled_scores.quantile(upper_quantile, dim=1) 
        ], dim=1) 
        
        return q_hat
        
    def train_hopcpt(self, data_loader, lr=0.001, num_epochs=3000):
        input_dim, hidden_dim, output_dim = next(iter(data_loader))[0].shape[1], 128, 16
        
        if self.encoding_model == None:
            self.encoding_model = EncodingNetwork(input_dim, hidden_dim, output_dim).to(self._device)
        if self.Hopfield_model == None:
            self.Hopfield_model = HopfieldAssociation(output_dim+1, hidden_dim).to(self._device)
            
        loss_fn = HopCPTLoss()
        optimizer = optim.Adam(
            list(self.encoding_model.parameters()) + list(self.Hopfield_model.parameters()), lr=lr
        )
        
        dataset = data_loader.dataset.dataset
        T = len(dataset)
        dataset_t = Dataset_T(dataset, range(T))
        data_loader_with_t = DataLoader(dataset_t, batch_size=data_loader.batch_size)

        with tqdm(total=num_epochs, desc="Epoch") as _tqdm:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for (x, y), t in data_loader_with_t:
                    x, y, t = x.to(self._device), y.to(self._device), t.to(self._device)
                    optimizer.zero_grad()
                    encoded_Z_t = self.encoding_model(x, t, T)
                    errors = self._model(x) - y
                    A = self.Hopfield_model(encoded_Z_t)

                    loss = loss_fn(errors, A)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.6f}")
    
    def _compute_attention_weights(self, X_pass, X_current):
        with torch.no_grad():
            t1 = len(X_pass)
            t2 = len(X_current)
            Z_pass = self.encoding_model(X_pass, torch.tensor(range(t1), device=self._device), t1)
            Z_current = self.encoding_model(X_current, torch.tensor([t1] * (t2), device=self._device), t1)
            query = self.Hopfield_model.W_q(Z_current)
            key = self.Hopfield_model.W_k(Z_pass)
            
            scores = torch.matmul(query, key.T)
            attention_weights = torch.softmax(self.Hopfield_model.beta * scores, dim=-1)
            
        # attention_weights = torch.ones_like(attention_weights)
        return attention_weights.squeeze(1)
    
    def _split_dataloader(self, dataloader, ratio):
        dataset = dataloader.dataset 
        train_size = int(len(dataset) * ratio)

        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))

        batch_size = dataloader.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


T_co = TypeVar('T_co', covariant=True)
class Dataset_T(Dataset[T_co]):
    
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]], idx

    def __len__(self):
        return len(self.indices)