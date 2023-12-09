
# The reference repository is https://github.com/aangelopoulos/conformal_classification



import numpy as np
import torch

from deepcp.classification.scores.base import DaseScoreFunction

class RAPS(DaseScoreFunction):
    def __init__(self, class_num = 100, penalty = 0, kreg = 0,randomized=True):
        super(RAPS, self).__init__()
        self.__randomized = randomized
        self.__lambda = penalty
        if type(penalty) is not float:
            self.penalties = penalty
        else:
            self.penalties = torch.zeros(class_num)
            self.penalties[kreg:] += penalty
        self.penalties_cumsum =  torch.cumsum(self.penalties)
    def __call__(self, probabilities, y):

        # sorting probabilities
        I, ordered, cumsum = self.__sort_sum(probabilities)
        idx = torch.where(I == y)[0]
        tau_nonrandom = cumsum[idx]

        if not self.__randomized:
            return tau_nonrandom + self.penalties[:idx].sum()

        U = np.random.random()
        if idx == torch.tensor(0):
                return U * tau_nonrandom + self.penalties[0]
        else:
            if idx[0] == cumsum.shape[0]:
                return U * ordered[idx] + cumsum[ idx - 1] + self.penalties[:idx].sum()
            else:
                return U * ordered[idx] + cumsum[ idx - 1] + self.penalties[:idx+1].sum()

    def predict(self, probabilities):
        I, ordered, cumsum = self.__sort_sum(probabilities)
        U = torch.rand(probabilities.shape[0])
        if self.__randomized:
            ordered_scores = cumsum - ordered * U + self.penalties_cumsum
        else:
            ordered_scores = cumsum + self.penalties_cumsum

        return ordered_scores[torch.sort(I,descending= False)[1]]

    def __sort_sum(self,probabilities):

        #ordered: the ordered probabilities in descending order
        #indices: the rank of ordered probabilities in descending order
        ordered,indices = torch.sort(probabilities,descending= True)
        # the accumulation of sorted probabilities
        cumsum = torch.cumsum(ordered,dim=0)
        return indices, ordered, cumsum

