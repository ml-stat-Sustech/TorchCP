

from deepcp.classification.scores.base import DaseScoreFunction

class THR(DaseScoreFunction):
    def __init__(self) -> None:
        pass
    
    
    def __call__(self,probabilities,y):
        return 1-probabilities[y]
    
    def predict(self,probabilities):
        return 1-probabilities