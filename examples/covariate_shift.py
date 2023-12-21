# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import json
import os

import torch
import torch.nn as nn

from dataset import build_dataset
from torchcp.classification.predictors import WeightedPredictor
from torchcp.classification.scores import THR
from torchcp.utils import fix_randomness

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Covariate shift')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = 'CLIP_ViTB16'
    from clip import load as clipload
    from clip import tokenize

    clip, preprocess = clipload('ViT-B/16', model_device)

    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data", "imagenet")
    with open(os.path.join(data_dir, 'human_readable_labels.json')) as f:
        readable_labels = json.load(f)


    class CLIPModel(nn.Module):
        def __init__(self, clip_model, readable_labels, model_device) -> None:
            super().__init__()
            text_inputs = torch.cat([tokenize(f"a photo of a {c}") for c in readable_labels]).to(model_device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_inputs)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_device)
            self.clip_model = clip_model

        def forward(self, x_batch):
            image_features = self.clip_model.encode_image(x_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return (100.0 * image_features @ self.text_features.T)


    model = CLIPModel(clip, readable_labels, model_device)

    ##################################
    # Invalid prediction sets
    ##################################
    src_dataset = build_dataset("imagenet", preprocess)
    tar_dataset = build_dataset("imagenetv2", preprocess)
    src_dataset, _ = torch.utils.data.random_split(src_dataset, [1000, 49000])
    tar_dataset, _ = torch.utils.data.random_split(tar_dataset, [1000, 9000])

    cal_data_loader = torch.utils.data.DataLoader(src_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(tar_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    score_function = THR()
    alpha = 0.1


    # predictors = StandardPredictor(score_function, model)
    # predictors.calibrate(cal_data_loader, alpha)

    # # test examples
    # print("Testing examples...")
    # print(predictors.evaluate(test_data_loader))

    ##################################
    # Invalid prediction sets
    ##################################

    class ImageEncoder(nn.Module):
        def __init__(self, model):
            super(ImageEncoder, self).__init__()
            self.clip = model

        def forward(self, x_bs):
            image_features = clip.encode_image(x_bs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features


    image_encoder = ImageEncoder(clip.eval().to(model_device))

    predictor = WeightedPredictor(score_function, model, image_encoder)
    predictor.calibrate(cal_data_loader, alpha)

    # # test examples
    print("Testing examples...")
    print(predictor.evaluate(test_data_loader))
