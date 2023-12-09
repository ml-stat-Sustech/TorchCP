
# The reference repository is https://github.com/aangelopoulos/conformal_classification



import numpy as np
import torch

from deepcp.classification.scores.aps import APS

class SAPS(APS):
    def __init__(self, penalty = 0,randomized=True):
        """

        :kreg : the rank of regularization [0,labels_num]
        """
        super(APS, self).__init__()
        self.__randomized = randomized
        self.__penalty = penalty

    def __call__(self, probabilities, y):

        # sorting probabilities
        I, ordered, cumsum = self.__sort_sum(probabilities)
        idx = torch.where( I == y )[0]
        if not self.__randomized:
            return self.__penalty*idx  + ordered[0]
        else:
            U = torch.rand(1)[0]
            if idx == torch.tensor(0):
                return U * cumsum[idx]
            else:
                return self.__penalty*(idx-U)  + ordered[0]



    def predict(self, probabilities):
        I, ordered, _ = self.__sort_sum(probabilities)
        ordered[1:] = 0
        cumsum = torch.cumsum(ordered,dim=0)
        U = torch.rand(probabilities.shape[0])
        if self.__randomized:
            ordered_scores = cumsum - ordered * U
        else:
            ordered_scores = cumsum
        return ordered_scores[torch.sort(I,descending= False)[1]]

