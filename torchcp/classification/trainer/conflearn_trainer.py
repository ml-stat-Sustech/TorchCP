# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.optim as optim


class UniformMatchingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = len(x)
        if batch_size == 0:
            return 0
        # Soft-sort the input
        x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
        i_seq = torch.arange(1.0,1.0+batch_size,device=device)/(batch_size)
        out = torch.max(torch.abs(i_seq - x_sorted))
        return out
  

class ConfLearnTrainer:

    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion_pred_loss_fn=torch.nn.CrossEntropyLoss(),
                 mu: float=0.2,
                 alpha: float=0.1,
                 device: torch.device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.criterion_pred_loss_fn = criterion_pred_loss_fn
        self.mu = mu
        self.alpha = alpha

        self.criterion_scores = UniformMatchingLoss()
        self.layer_prob = torch.nn.Softmax(dim=1)

    def compute_loss_scores(self, y_train_pred, y_train_batch, alpha=0.1):
        train_proba = self.layer_prob(y_train_pred)
        train_scores, train_sizes = compute_scores_diff(train_proba, y_train_batch, alpha=alpha)
        train_loss_scores = self.criterion_scores(train_scores)
        return train_loss_scores

    def calculate_loss(self, output, target, Z_batch):
        idx_ce = torch.where(Z_batch == 0)[0]
        loss_ce = self.criterion_pred_loss_fn(output[idx_ce], target[idx_ce])

        loss_scores = torch.tensor(0.0, self.device)
        Z_groups = torch.unique(Z_batch)
        n_groups = torch.sum(Z_groups > 0)
        for z in Z_groups:
            idx_z = torch.where(Z_batch == z)[0]
            loss_scores_z = self.compute_loss_scores(output[idx_z], target[idx_z], alpha=self.alpha)
            loss_scores += loss_scores_z
        loss_scores /= n_groups

        loss = loss + loss_scores * self.mu
        return loss

    def train_epoch(self, train_loader):
        self.model.train()

        for X_batch, Y_batch, Z_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            Z_batch = Z_batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(X_batch)

            # Calculate loss
            loss = self.calculate_loss(output, Y_batch, Z_batch)

            loss.backward()
            self.optimizer.step()

    def validate(self, val_loader):
        pass

    def train(self, num_epochs, train_loader, val_loader=None):
        lr_milestones = [int(num_epochs*0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.1)

        for epoch in range(num_epochs):
            self.train_epoch(train_loader)
            
            scheduler.step()
            self.model.eval()

            if val_loader is not None:
                self.validate(val_loader)