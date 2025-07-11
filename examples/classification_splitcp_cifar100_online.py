import torch
import torchvision
import torchvision.transforms as trn
from transformers import set_seed
from tqdm import tqdm

from examples.utils import get_dataset_dir
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import LAC

set_seed(seed=0)

#######################################
# Dataset preparation
#######################################
transform = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean=[0.5071, 0.4867, 0.4408],
                  std=[0.2675, 0.2565, 0.2761])
])

dataset = torchvision.datasets.CIFAR100(
    root=get_dataset_dir(),
    train=False,
    download=True,
    transform=transform
)
cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [5000, 5000])  

#######################################
# Model preparation
#######################################
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

cal_logits, cal_labels = [], []
for x, y in cal_dataset:
    cal_logits.append(model(x.unsqueeze(0).to(device)))
    cal_labels.append(y)
    
cal_logits = torch.cat(cal_logits, dim=0)
cal_labels = torch.tensor(cal_labels)


test_logits, test_labels = [], []
for x, y in test_dataset:
    test_logits.append(model(x.unsqueeze(0).to(device)))
    test_labels.append(y)

test_logits = torch.cat(test_logits, dim=0)
test_labels = torch.tensor(test_labels)

#######################################
# Online Conformal Prediction
#######################################
alpha = 0.1

predictor = SplitPredictor(score_function=LAC(), model=model, alpha=alpha, device=device)

cover_count = 0
total = 0
set_size_sum = 0

for i in tqdm(range(len(test_dataset))):
    predictor.calculate_threshold(cal_logits, cal_labels, alpha)
    
    x, y = test_dataset[i]
    logits = model(x.unsqueeze(0).to(device))
    
    
    pred_set = predictor.predict_with_logits(logits)[0] 
    covered = 1 if pred_set[y] else 0

    cover_count += int(covered)
    set_size_sum += pred_set.sum()
    total += 1

    cal_logits = torch.cat((cal_logits, logits), dim=0)
    cal_labels = torch.cat((cal_labels, torch.tensor([y])), dim=0)

coverage_rate = cover_count / total
average_set_size = set_size_sum / total

print(f"Online CP Coverage Rate: {coverage_rate:.4f}")
print(f"Online CP Average Set Size: {average_set_size:.4f}")
