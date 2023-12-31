
import torch.nn as nn


def build_regression_model(model_name):
    if model_name == "NonLinearNet":
        class NonLinearNet(nn.Module):
            def __init__(self, in_shape, out_shape, hidden_size, dropout):
                super(NonLinearNet, self).__init__()
                self.hidden_size = hidden_size
                self.in_shape = in_shape
                self.out_shape = out_shape
                self.dropout = dropout
                self.base_model = nn.Sequential(
                    nn.Linear(self.in_shape, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.out_shape),
                )

            def forward(self, x):
                return self.base_model(x)

        return NonLinearNet
    else:
        raise NotImplementedError



