
from abc import ABCMeta, abstractmethod


class DaseScoreFunction(object):
    """
    Abstract base class for all score functions.

    """
    __metaclass__ = ABCMeta
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self,probabilities, y ):
        """Virtual method to compute scores for a data pair (x,y).

        :param probabilities: the model's output probabilities for an input.
        :y : the label.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, probabilities):
        """Virtual method to compute scores of all labels for input x.

        :param probabilities: the model's output probabilities for an input.
        """
        raise NotImplementedError