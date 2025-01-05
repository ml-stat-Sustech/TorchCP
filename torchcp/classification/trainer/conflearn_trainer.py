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
from torchsort import soft_rank, soft_sort

REG_STRENGTH = 0.1
B = 50

class UniformMatchingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = len(x)
        if batch_size == 0:
            return 0
        x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
        i_seq = torch.arange(1.0, 1.0 + batch_size, device=x.device)/(batch_size)
        out = torch.max(torch.abs(i_seq - x_sorted))
        return out
    

def soft_indicator(x, a, b=B):
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    out = torch.sigmoid(b * (x - a + 0.5)) - (torch.sigmoid(b * (x - a - 0.5)))
    out = out / (sigmoid(b * (0.5)) - (sigmoid(b * (-0.5))) )
    return out


def soft_indexing(z, rank):
    n = len(rank)
    K = z.shape[1]
    I = torch.tile(torch.arange(K, device=z.device), (n, 1))
    weight = soft_indicator(I.T, rank).T
    weight = weight * z
    return weight.sum(dim=1)


def accuracy_point(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc*100
  

class ConfLearnTrainer:

    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion_pred_loss_fn=torch.nn.CrossEntropyLoss(),
                 mu: float=0.2,
                 alpha: float=0.1,
                 device: torch.device=None):
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

    def compute_scores_diff(self, proba_values, Y_values, alpha=0.1):
        n, K = proba_values.shape
        proba_values = proba_values + 1e-6 * torch.rand(proba_values.shape, dtype=float, device=self.device)
        proba_values = proba_values / torch.sum(proba_values,1)[:,None]
        ranks_array_t = soft_rank(-proba_values, regularization_strength=REG_STRENGTH)-1
        prob_sort_t = -soft_sort(-proba_values, regularization_strength=REG_STRENGTH)
        Z_t = prob_sort_t.cumsum(dim=1)

        ranks_t = torch.gather(ranks_array_t, 1, Y_values.reshape(n,1)).flatten()
        prob_cum_t = soft_indexing(Z_t, ranks_t)
        prob_final_t = soft_indexing(prob_sort_t, ranks_t)
        scores_t = 1.0 - prob_cum_t + prob_final_t * torch.rand(n,dtype=float,device=self.device)

        return scores_t

    def compute_loss_scores(self, y_train_pred, y_train_batch, alpha=0.1):
        train_proba = self.layer_prob(y_train_pred)
        train_scores = self.compute_scores_diff(train_proba, y_train_batch, alpha=alpha)
        train_loss_scores = self.criterion_scores(train_scores)
        return train_loss_scores

    def calculate_loss(self, output, target, Z_batch, training=True):
        if training:
            idx_ce = torch.where(Z_batch == 0)[0]
            loss_ce = self.criterion_pred_loss_fn(output[idx_ce], target[idx_ce])
        else:
            loss_ce = self.criterion_pred_loss_fn(output, target)

        loss_scores = torch.tensor(0.0, device=self.device)
        Z_groups = torch.unique(Z_batch)
        n_groups = torch.sum(Z_groups > 0)
        for z in Z_groups:
            if z > 0:
                idx_z = torch.where(Z_batch == z)[0]
                loss_scores_z = self.compute_loss_scores(output[idx_z], target[idx_z], alpha=self.alpha)
                loss_scores += loss_scores_z
        loss_scores /= n_groups

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
              checkpoint_path: str=None, 
              num_epochs=10):
        if save_model:
            if checkpoint_path is None:
                raise("Output checkpoint name file is needed.")
            
        best_loss = None
        best_acc = None

        lr_milestones = [int(num_epochs*0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.1)

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
                    self.save_checkpoint(epoch, checkpoint_path+"_loss")
                
                # Early stopping by accuracy
                save_checkpoint = True if best_acc is not None and best_acc < epoch_acc_val else False
                best_acc = epoch_acc_val if best_acc is None or best_acc < epoch_acc_val else best_acc
                if save_checkpoint:
                    self.save_checkpoint(epoch, checkpoint_path+"_acc")

        if save_model:
            self.save_checkpoint(epoch, checkpoint_path+"_final")
        

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
                y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        return y_pred