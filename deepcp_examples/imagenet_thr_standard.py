# Copyright (c) 2023-present, (Nan).
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
from tqdm import tqdm
import pickle 

import torch
import torchvision
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.nn.functional import softmax

from deepcp.classification.scores import THR  
from deepcp.classification.predictor import StandardPredictor,ClassWisePredictor
from deepcp.classification.utils.metircs import Metrics
from deepcp.utils import fix_randomness


fix_randomness(seed = 0)


model_name = 'ResNet101'
fname = ".cache/"+model_name+".pkl"
if os.path.exists(fname):
    with open(fname, 'rb') as handle:
        dataset =  pickle.load(handle)

else:
    # load dataset
    transform = trn.Compose([trn.Resize(256),
                                    trn.CenterCrop(224),
                                    trn.ToTensor(),
                                    trn.Normalize(mean=[0.485, 0.456, 0.406],
                                                std =[0.229, 0.224, 0.225])
                                    ])
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir,"data")
    dataset = dset.ImageFolder(data_dir+"/imagenet/val", 
                                        transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 320, shuffle=False, pin_memory=True)

    # load model
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)


    logits_list = []
    labels_list = []
    with torch.no_grad():
        for examples in tqdm(data_loader):
            tmp_x, tmp_label = examples[0], examples[1]            
            tmp_logits = model(tmp_x)
            logits_list.append(tmp_logits)
            labels_list.append(tmp_label)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    dataset = torch.utils.data.TensorDataset(logits, labels.long()) 
    with open(fname, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

cal_data, val_data = torch.utils.data.random_split(dataset, [25000, 25000])

cal_logits = torch.stack([sample[0] for sample in cal_data])
cal_labels = torch.stack([sample[1] for sample in cal_data])
cal_probailities =  softmax(cal_logits,dim=1)

test_logits = torch.stack([sample[0] for sample in val_data])
test_labels = torch.stack([sample[1] for sample in val_data])
test_probailities =  softmax(test_logits,dim=1)

thr_score_function = THR()
alpha = 0.1
predictor = ClassWisePredictor(thr_score_function)
predictor.fit(cal_probailities, cal_labels, alpha)

# test examples
print("testing examples...")
prediction_sets = []
for index,ele in enumerate(test_probailities):
    prediction_set  = predictor.predict(ele)
    prediction_sets.append(prediction_set)

print("Evaluating prediction sets...")
metrics = Metrics(["coverage_rate","average_size"])
print(metrics.compute(prediction_sets,test_labels))
