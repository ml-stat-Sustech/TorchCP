# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

from examples.utils import get_others_dir
from llm_utils import *
from torchcp.llm.predictor import ConformalLM

if __name__ == "__main__":

    dataset_name = "trivia_qa"
    model_path = "meta-llama/Llama-2-7b-hf"
    set_seed(2025)

    output_path = os.path.join(get_others_dir(), f"{dataset_name}_results.npz")
    if not os.path.exists(output_path):
        preprocess_data(dataset_name, model_path, output_path)

    scaling_type = 'platt'
    scale_kwargs = {}

    data = np.load(output_path)
    labels = torch.from_numpy(data['labels'])
    scores = torch.from_numpy(data['scores'])
    diversity = torch.from_numpy(data['diversity'])

    N_train = 2000 if scaling_type is not None else 0
    training_idx, tuning_idx, cal_idx, val_idx = split_indices(len(labels), N_train, 0.3, 0.3)

    conformal_llm = ConformalLM(
        epsilons=[0.2],
        scaling_type=scaling_type,
        scale_kwargs=scale_kwargs,
        set_score_function_name='geo',
        rejection=True
    )
    if scaling_type is not None:
        conformal_llm.scaling(scores[training_idx], labels[training_idx])
    conformal_llm.tuning(scores[tuning_idx], diversity[tuning_idx], labels[tuning_idx])
    conformal_llm.calibrate_configs(
        scores[cal_idx],
        diversity[cal_idx],
        labels[cal_idx],
        alpha=0.05
    )
    results = conformal_llm.evaluate(
        scores[val_idx],
        diversity[val_idx],
        labels[val_idx]
    )

    # Calculate and print statistics
    print("\nResults:")
    print("-" * 50)
    print(f"Average Loss: {results['avg_losses'][0]:.4f}")
    print(f"Average Set Size: {results['avg_size'][0]:.4f}")
