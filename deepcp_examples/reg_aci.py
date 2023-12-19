import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from deepcp.utils import fix_randomness
from deepcp.regression.predictor import SplitPredictor,CQR
from deepcp.regression.utils.metrics import Metrics
from deepcp.regression.loss import QuantileLoss

from utils import build_reg_data
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
fix_randomness(seed=2)

X,y = build_reg_data(data_name="synthetic")
num_examples = X.shape[0]

T0 =  int(num_examples * 0.4)  

train_dataset = TensorDataset(torch.from_numpy(X[:T0,:]),torch.from_numpy(y[:T0]))
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)


class NonLinearNet(nn.Module):
    def __init__(self, in_shape, hidden_size,dropout ):
        super(NonLinearNet, self).__init__()
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = 2
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
    
alpha = 0.1
quantiles = [alpha/2, 1-alpha/2]
model = NonLinearNet(X.shape[1], 64, 0.5).to(device)
criterion = QuantileLoss(quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train Model
epochs = 10
for epoch in tqdm(range(epochs)):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader): 
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
predictor = CQR(model, device)


y_list = []
x_list = []
predict_list = []
for i in range(num_examples - T0):
    with torch.no_grad():
        cal_dataset = TensorDataset(torch.from_numpy(X[i:(T0+i),:]),torch.from_numpy(y[i:(T0+i)]))
        cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
        predictor.calibrate(cal_data_loader, alpha)
        tmp_x =  torch.from_numpy(X[(T0+i),:])
        tmp_prediction_intervals = predictor.predict(tmp_x)
        y_list.append(y[T0+i])
        x_list.append(tmp_x)
        predict_list.append(tmp_prediction_intervals)
        
predicts = torch.stack(predict_list, dim=0)
test_y = torch.Tensor(y_list)
x = torch.cat(x_list).float()

metrics = Metrics()
print("Etestuating prediction sets...")
print(f"Coverage_rate: {metrics('coverage_rate')(predicts, test_y)}.")
print(f"Average_size: {metrics('average_size')(predicts, test_y)}.")


    


