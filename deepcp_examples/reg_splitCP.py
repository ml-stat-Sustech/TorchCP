import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F


from deepcp.utils import fix_randomness
from deepcp.regression.predictor import SplitPredictor
from deepcp.regression.scores import ABS
from deepcp.regression.utils.metrics import Metrics



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
fix_randomness(seed=2)
epsilon = 0.01


# 生成非线性数据
x = torch.linspace(-np.pi*2, np.pi*2, 200).reshape(-1, 1)
y_truth = 0.1*x**2 + torch.cos(x)
y = y_truth  + torch.randn(x.size()) * epsilon

dataset = TensorDataset(x, y)
train_dataset,cal_dataset,test_dataset = torch.utils.data.random_split(dataset, [80,60,60])

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, pin_memory=True)
cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)


class NonLinearNet(nn.Module):
    def __init__(self):
        super(NonLinearNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NonLinearNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 1000
for epoch in range(epochs):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader): 
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
score_function = ABS()
predictor = SplitPredictor(score_function, model, device)
predictor.calibrate(cal_data_loader, 0.1)
predicts =  predictor.predicts
labels =  predictor.labels
x =  predictor.x
plt.scatter(x.cpu(),labels.cpu())
plt.scatter(x.cpu(),predicts.cpu())
plt.savefig(".cache/reg.jpg")

with torch.no_grad():
    for  examples in test_data_loader:
        tmp_x, tmp_labels = examples[0].to(device), examples[1]
        tmp_prediction_intervals = predictor.predict(tmp_x)
tmp_prediction_intervals = tmp_prediction_intervals.cpu().numpy()
print(tmp_prediction_intervals.shape)
plt.scatter(tmp_x.cpu().numpy(), tmp_labels.cpu().numpy())
plt.fill_between(tmp_x.cpu().numpy().reshape(-1), tmp_prediction_intervals[:,0], tmp_prediction_intervals[:,1], color='blue', alpha=0.3)

plt.savefig(".cache/reg.jpg")

metrics = Metrics()
print("Etestuating prediction sets...")
print(f"Coverage_rate: {metrics('coverage_rate')(tmp_prediction_intervals, tmp_labels)}.")
print(f"Average_size: {metrics('average_size')(tmp_prediction_intervals, tmp_labels)}.")

# print(predictor.evaluate(test_data_loader))
    


