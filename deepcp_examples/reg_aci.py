import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from deepcp.regression import Metrics
from deepcp.regression.loss import QuantileLoss
from deepcp.regression.predictor import ACI
from deepcp.utils import fix_randomness
from utils import build_reg_data, build_regression_model

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
fix_randomness(seed=2)

X, y = build_reg_data(data_name="synthetic")
num_examples = X.shape[0]

T0 = int(num_examples * 0.4)

train_dataset = TensorDataset(torch.from_numpy(X[:T0, :]), torch.from_numpy(y[:T0]))
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

alpha = 0.1
quantiles = [alpha / 2, 1 - alpha / 2]
model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
criterion = QuantileLoss(quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in tqdm(range(epochs)):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

predictor = ACI(model, 0.0001)

test_y = torch.from_numpy(y[T0:num_examples]).to(device)
predicts = torch.zeros((num_examples - T0, 2)).to(device)
for i in range(num_examples - T0):
    with torch.no_grad():
        cal_dataset = TensorDataset(torch.from_numpy(X[i:(T0 + i), :]), torch.from_numpy(y[i:(T0 + i)]))
        cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
        predictor.calibrate(cal_data_loader, alpha)
        tmp_x = torch.from_numpy(X[(T0 + i), :])
        if i == 0:
            tmp_prediction_intervals = predictor.predict(tmp_x)
        else:
            tmp_prediction_intervals = predictor.predict(tmp_x, test_y[i - 1], predicts[i - 1])
        predicts[i, :] = tmp_prediction_intervals

metrics = Metrics()
print("Etestuating prediction sets...")
print(f"Coverage_rate: {metrics('coverage_rate')(predicts, test_y)}.")
print(f"Average_size: {metrics('average_size')(predicts, test_y)}.")
