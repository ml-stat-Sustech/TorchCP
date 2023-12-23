import torch
import torch.nn as nn
import torch.nn.functional as F


def build_DomainDetecor(in_dim, out_dim, device):
    return MidFNN(in_dim, out_dim).to(device)


class FNN(nn.Module):
    def __init__(self, n_in, n_out, n_hiddens, n_layers):
        super().__init__()

        models = []
        for i in range(n_layers):
            n = n_in if i == 0 else n_hiddens
            models.append(nn.Linear(n, n_hiddens))
            models.append(nn.ReLU())
            models.append(nn.Dropout(0.5))
        models.append(nn.Linear(n_hiddens if n_hiddens is not None else n_in, n_out))
        self.model = nn.Sequential(*models)

    def forward(self, x, training=False):
        if training:
            self.model.train()
        else:
            self.model.eval()
        logits = self.model(x)
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, -1)
        return probs


class Linear(FNN):
    def __init__(self, n_in, n_out, n_hiddens=None, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=0, path_pretrained=path_pretrained)


class SmallFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1, path_pretrained=path_pretrained)


class MidFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2, path_pretrained=path_pretrained)


class BigFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500, path_pretrained=None):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4, path_pretrained=path_pretrained)


class IW(nn.Module):
    """
    Compute the importance weight
    """

    def __init__(self, domain_detecor):
        super().__init__()

        self.domain_detecor = domain_detecor

    def forward(self, x_batch):
        prob = self.domain_detecor(x_batch)
        if prob.shape[1] == 1:
            return prob / (1 - prob)
        else:
            return prob[:, 1] / prob[:, 0]
