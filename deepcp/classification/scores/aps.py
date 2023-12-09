
# The reference repository is https://github.com/aangelopoulos/conformal_classification



import numpy as np
import torch

from deepcp.classification.scores.base import DaseScoreFunction

class APS(DaseScoreFunction):
    def __init__(self, penalty = 0, kreg = 0,randomized=True):
        """

        :kreg : the rank of regularization [0,labels_num]
        """
        super(APS, self).__init__()
        self.__randomized = randomized
        self.__penalty = penalty
        self.__kreg = kreg

    def __call__(self, probabilities, y):

        # sorting probabilities
        I, ordered, cumsum = self.__sort_sum(probabilities)
        idx = torch.where(I == y)[0]
        if not self.__randomized:
            return cumsum[idx] + torch.maximum(self.__penalty * (idx+1 - self.__kreg), torch.tensor(0))
        else:
            U = torch.rand(1)
            return U * ordered[idx] + cumsum[idx - 1] + torch.maximum(self.__penalty * (idx+1 - self.__kreg), torch.tensor(0))



    def predict(self, probabilities):
        I, ordered, cumsum = self.__sort_sum(probabilities)
        U = torch.rand(probabilities.shape[0])
        reg = torch.maximum(self.__penalty * ( torch.arange(1,probabilities.shape[0]+1) - self.__kreg), torch.zeros(probabilities.shape[0]))
        if self.__randomized:
            ordered_scores = cumsum - ordered * U + reg
        else:
            ordered_scores = cumsum + reg

        return ordered_scores[torch.sort(I,descending= False)[1]]

    def __sort_sum(self,probabilities):

        #ordered: the ordered probabilities in descending order
        #indices: the rank of ordered probabilities in descending order
        ordered,indices = torch.sort(probabilities,descending= True)
        # the accumulation of sorted probabilities
        cumsum = torch.cumsum(ordered,dim=0)
        return indices, ordered, cumsum

