import numpy as np
import torch


class ProbabilityAccumulator:
    def __init__(self, prob):
        self._device = prob.device
        self.n, self.K = prob.shape
        self.order = torch.argsort(-prob, dim=1).to(self._device)
        self.ranks = torch.empty_like(self.order).to(self._device)
        self.ranks.scatter_(1, self.order, torch.arange(self.K, device=self._device).expand(self.n, -1))
        self.prob_sort = -torch.sort(-prob, dim=1).values
        # self.Z = torch.round(torch.cumsum(self.prob_sort, dim=1) * 1e9) / 1e9
        self.Z = torch.cumsum(self.prob_sort, dim=1)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        # TODO: document shape of alpha for vectorized impl. Input should be shape n x 1.
        # TODO: should make this neater.

        L = torch.argmax((self.Z >= 1.0 - alpha).float(), dim=1).flatten()
        if epsilon is not None:
            Z_excess = self.Z[torch.arange(self.n), L] - (1.0 - alpha).flatten()
            p_remove = Z_excess / self.prob_sort[torch.arange(self.n), L]
            remove = epsilon <= p_remove
            for i in torch.where(remove)[0]:
                if not allow_empty:
                    L[i] = torch.maximum(torch.tensor(0), L[i] - 1)
                else:
                    L[i] = L[i] - 1
        
        S = [self.order[i, torch.arange(0, L[i] + 1)] for i in range(self.n)]
        return (S)

    def calibrate_scores(self, Y, epsilon=None):
        "1-score"
        n2 = Y.shape[0]
        ranks = self.ranks[torch.arange(n2), Y]
        prob_cum = self.Z[torch.arange(n2), ranks]
        prob = self.prob_sort[torch.arange(n2), ranks]
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += torch.mul(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = torch.minimum(alpha_max, torch.tensor(1.0))
        return alpha_max