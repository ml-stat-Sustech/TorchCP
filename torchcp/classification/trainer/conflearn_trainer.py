# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.optim as optim

class ConfLearnTrainer:

    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model.to(self.device)
        
        self.optimizer = optimizer

    def calculate_loss(self, output, target):
        pass

    def train_epoch(self, train_loader):
        self.model.train()

        for X_batch, Y_batch, Z_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            Z_batch = Z_batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(X_batch)

            # Calculate loss
            loss = self.calculate_loss(output, Y_batch)

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