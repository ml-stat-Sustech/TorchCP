import numpy as np
from tqdm import tqdm


class ProbabilityAccumulator:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        # self.epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.Z = np.round(self.prob_sort.cumsum(axis=1), 9)

    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        # TODO: document shape of alpha for vectorized impl. Input should be shape n x 1.
        # TODO: should make this neater.
        L = np.argmax(self.Z >= 1.0 - alpha, axis=1).flatten()
        if epsilon is not None:
            # Get the amount that summing to above 1-alpha exceeds 1-alpha.
            Z_excess = np.array([self.Z[i, L[i]] for i in range(self.n)]) - (1.0 - alpha).flatten()
            # Probability we remove the last element is the gap divided by the total sum.
            p_remove = Z_excess / np.array([self.prob_sort[i, L[i]] for i in range(self.n)])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1

        # Return prediction set
        S = [self.order[i, np.arange(0, L[i] + 1)] for i in range(self.n)]
        return (S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([self.ranks[i, Y[i]] for i in range(n2)])
        prob_cum = np.array([self.Z[i, ranks[i]] for i in range(n2)])
        prob = np.array([self.prob_sort[i, ranks[i]] for i in range(n2)])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)  # Uniform sampling step.
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max