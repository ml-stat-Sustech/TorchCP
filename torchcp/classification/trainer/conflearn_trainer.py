# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torchcp.classification.loss import ConfLearnLoss


def accuracy_point(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc * 100


class ConfLearnTrainer:

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion_pred_loss_fn=torch.nn.CrossEntropyLoss(),
                 mu: float = 0.2,
                 alpha: float = 0.1,
                 device: torch.device = None):
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.criterion_pred_loss_fn = criterion_pred_loss_fn
        self.conformal_loss_fn = ConfLearnLoss(device, alpha)
        self.mu = mu
        self.alpha = alpha


    def calculate_loss(self, output, target, Z_batch, training=True):
        if training:
            idx_ce = torch.where(Z_batch == 0)[0]
            loss_ce = self.criterion_pred_loss_fn(
                output[idx_ce], target[idx_ce])
        else:
            loss_ce = self.criterion_pred_loss_fn(output, target)

        loss_scores = self.conformal_loss_fn(output, target, Z_batch)

        loss = loss_ce + loss_scores * self.mu
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

    @torch.no_grad()
    def validate(self, val_loader):
        loss_val = 0
        acc_val = 0

        for X_batch, Y_batch, Z_batch in val_loader:
            output = self.model(X_batch)
            loss = self.calculate_loss(output, Y_batch, Z_batch)
            acc = accuracy_point(output, Y_batch)

            loss_val += loss.item()
            acc_val += acc.item()

        loss_val /= len(val_loader)
        acc_val /= len(val_loader)

        return loss_val, acc_val

    def train(self,
              train_loader,
              val_loader=None,
              early_stopping=True,
              save_model=True,
              checkpoint_path: str = None,
              num_epochs=10):
        if save_model:
            if checkpoint_path is None:
                raise ("Output checkpoint name file is needed.")

        best_loss = None
        best_acc = None

        lr_milestones = [int(num_epochs * 0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.1)

        for epoch in tqdm(range(num_epochs)):

            self.train_epoch(train_loader)

            scheduler.step()
            self.model.eval()

            if val_loader is not None:
                epoch_loss_val, epoch_acc_val = self.validate(val_loader)

            if early_stopping:
                # Early stopping by loss
                save_checkpoint = True if best_loss is not None and best_loss > epoch_loss_val else False
                best_loss = epoch_loss_val if best_loss is None or best_loss > epoch_loss_val else best_loss
                if save_checkpoint:
                    self.save_checkpoint(epoch, checkpoint_path + "_loss")

                # Early stopping by accuracy
                save_checkpoint = True if best_acc is not None and best_acc < epoch_acc_val else False
                best_acc = epoch_acc_val if best_acc is None or best_acc < epoch_acc_val else best_acc
                if save_checkpoint:
                    self.save_checkpoint(epoch, checkpoint_path + "_acc")

        if save_model:
            self.save_checkpoint(epoch, checkpoint_path + "_final")


    def save_checkpoint(self, epoch: int, save_path: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, load_path: str):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

    def predict(self, test_loader):
        y_pred_list = []
        with torch.no_grad():
            self.model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        return y_pred
