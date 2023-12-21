import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from torchcp.regression.loss import QuantileLoss
from torchcp.regression.predictors import CQR
from torchcp.regression.utils.metrics import Metrics
from torchcp.utils import fix_randomness
from utils import build_reg_data, build_regression_model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
fix_randomness(seed=2)

X, y = build_reg_data()
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split_index1 = int(len(indices) * 0.4)
split_index2 = int(len(indices) * 0.6)
part1, part2, part3 = np.split(indices, [split_index1, split_index2])

from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler()
scalerX = scalerX.fit(X[part1, :])

train_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part1, :])), torch.from_numpy(y[part1]))
cal_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part2, :])), torch.from_numpy(y[part2]))
test_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part3, :])), torch.from_numpy(y[part3]))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)
cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

alpha = 0.1
quantiles = [alpha / 2, 1 - alpha / 2]
model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
criterion = QuantileLoss(quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train Model
epochs = 100
for epoch in tqdm(range(epochs)):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

predictor = CQR(model)
predictor.calibrate(cal_data_loader, alpha)

y_list = []
x_list = []
predict_list = []
with torch.no_grad():
    for examples in test_data_loader:
        tmp_x, tmp_y = examples[0].to(device), examples[1]
        tmp_prediction_intervals = predictor.predict(tmp_x)
        y_list.append(tmp_y)
        x_list.append(tmp_x)
        predict_list.append(tmp_prediction_intervals)

predicts = torch.cat(predict_list).float().cpu()
test_y = torch.cat(y_list)
x = torch.cat(x_list).float()

metrics = Metrics()
print("Etestuating prediction sets...")
print(f"Coverage_rate: {metrics('coverage_rate')(predicts, test_y)}.")
print(f"Average_size: {metrics('average_size')(predicts, test_y)}.")

# print(predictors.evaluate(test_data_loader))
