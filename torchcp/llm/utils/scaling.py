"""Utilities for score scaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm


class BaseScaler:
    """Straight through scaler."""

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X


class LogisticRegression(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.dtype = dtype

        self.linear.weight.data = self.linear.weight.data.to(dtype)
        self.linear.bias.data = self.linear.bias.data.to(dtype)

    def forward(self, x):
        x = x.to(self.dtype)
        return torch.sigmoid(self.linear(x))


class PlattScaler(BaseScaler):
    """Parametric calibration using logistic regression on uncalibrated scores."""

    def __init__(self, **kwargs):
        super(PlattScaler, self).__init__()
        self.clf = LogisticRegression()
        self.optimizer = optim.SGD(self.clf.parameters(), lr=0.01)
        self.loss_fn = nn.BCELoss()
        self.num_epochs = kwargs.get('num_epochs', 100)

    def fit(self, X, y):
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            outputs = self.clf(X)
            loss = self.loss_fn(outputs, y.to(self.clf.dtype))
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.clf.eval()
        with torch.no_grad():
            scaled = self.clf(X.reshape(-1, 1))
        return scaled.reshape(X.shape)


class BinningScaler(BaseScaler):
    """Non-parametric equal-mass histogram regression on uncalibrated scores."""

    def __init__(self, n_bins=20):
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self.n_bins = n_bins

    def fit(self, X, y):
        X = X.reshape(-1)
        y = y.reshape(-1)

        # Split scores into equal mass bins.
        quantiles = torch.linspace(0, 1, self.n_bins + 1).to(X.dtype)
        bins = torch.quantile(X, quantiles)
        bin_ids = torch.searchsorted(bins[1:-1], X)

        # Count empirical accuracy in each bin.
        bin_true = torch.bincount(bin_ids, weights=y, minlength=len(bins))[:-1]
        bin_count = torch.bincount(bin_ids, minlength=len(bins))[:-1]
        bin_prob = bin_true / bin_count

        # Store bins and bin probs.
        self.bins = bins
        self.bin_prob = bin_prob

    def predict(self, X):
        bin_ids = torch.searchsorted(self.bins[1:-1], X.reshape(-1))
        return self.bin_prob[bin_ids].reshape(X.shape)


class PlattBinningScaler(BaseScaler):
    """Combined parametric + non-parametric calibration (Kumar et. al., 2019)"""

    def __init__(self, *args, n_bins=20, **kwargs):
        self.platt = PlattScaler(*args, **kwargs)
        self.binning = BinningScaler(n_bins=n_bins)

    def fit(self, X, y):
        # Split calibration set in two for scaling and binning.
        N = len(X) // 2
        X_scale, y_scale = X[:N], y[:N]
        X_bin, y_bin = X[N:], y[N:]

        # Fit Platt scaler.
        self.platt.fit(X_scale, y_scale)

        # Fit binning scaler on Platt scaled scores.
        self.binning.fit(self.platt.predict(X_bin), y_bin)

    def predict(self, X):
        return self.binning.predict(self.platt.predict(X))


class RecurrentScaler(BaseScaler, nn.Module):
    """RNN calibration of C given sequential scores."""

    def __init__(
            self,
            hidden_size=64,
            num_layers=2,
            num_iters=100,
            batch_size=128,
            dropout=0.0,
            target='set',
            verbose=True
    ):
        super(RecurrentScaler, self).__init__()
        if target not in ['set', 'item']:
            raise ValueError(f"target must be one of ['set', 'item'], but got {target}")
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.target = target
        self.verbose = verbose
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)
        self.hidden_to_output = nn.Linear(hidden_size, 1)

    def forward(self, X):
        out = self.rnn(X.unsqueeze(-1))[0]
        out = self.hidden_to_output(out).squeeze(-1)
        return out

    def fit(self, X, y):
        X = X.to(torch.float32)
        # If self.target = 'set', then we compute the set-based labels.
        if self.target == 'set':
            y = 1 - torch.cumprod(1 - y, axis=-1)

        dataset = TensorDataset(X, y)
        sampler = RandomSampler(
            dataset, num_samples=self.num_iters * self.batch_size)
        loader = DataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size)

        # Train loop.
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        if self.verbose:
            loader = tqdm(loader)
        for X_batch, y_batch in loader:
            out = self.forward(X_batch)
            loss = F.binary_cross_entropy_with_logits(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X = X.to(torch.float32)
        self.eval()
        with torch.no_grad():
            out = torch.sigmoid(self.forward(X))
        return out
