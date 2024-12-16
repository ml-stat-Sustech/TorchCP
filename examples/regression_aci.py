# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from transformers import set_seed

from examples.utils import build_reg_data
from torchcp.regression.loss import QuantileLoss
from torchcp.regression.predictor import ACIPredictor
from torchcp.regression.score import CQR
from torchcp.regression.utils import build_regression_model


def prepare_aci_dataset(X, y, train_ratio=0.5, batch_size=100):
    """
    Prepare datasets for Adaptive Conformal Inference.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        train_ratio (float): Ratio of training data
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: Training and test data loaders
    """
    # Split indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(len(indices) * train_ratio)
    train_indices, test_indices = np.split(indices, [split_index])

    # Scale features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[train_indices, :])

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[train_indices, :])),
        torch.from_numpy(y[train_indices])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[test_indices, :])),
        torch.from_numpy(y[test_indices])
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader


def run_aci_experiment(
        model,
        score_function,
        train_loader,
        test_loader,
        device,
        alpha=0.1,
        epochs=20,
        lr=0.01,
        gamma=0.005,
        verbose=True
):
    """
    Run Adaptive Conformal Inference experiment.
    
    Args:
        model (nn.Module): Neural network model
        score_function: Conformal score function
        train_loader: Training data loader
        test_loader: Test data loader
        device: Computing device
        alpha (float): Significance level
        epochs (int): Number of training epochs
        lr (float): Learning rate
        gamma (float): ACI hyperparameter
        verbose (bool): Whether to print detailed results
    
    Returns:
        dict: Evaluation results
    """
    print("\nRunning Adaptive Conformal Inference experiment...")

    # Setup training
    quantiles = [alpha / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize predictor
    predictor = ACIPredictor(score_function, model, gamma=gamma)

    # Train and evaluate
    predictor.train(
        train_dataloader=train_loader,
        alpha=alpha,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer
    )

    results = predictor.evaluate(test_loader)
    print(f"Results: {results}")

    return results


def main():
    """Main function to run ACI experiment."""
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed=1)

    # Hyperparameters
    epochs = 20
    alpha = 0.1
    gamma = 0.005
    hidden_size = 64
    dropout = 0.5

    print("Starting Adaptive Conformal Inference experiment...")

    # Load and prepare data
    X, y = build_reg_data(data_name="synthetic")
    train_loader, test_loader = prepare_aci_dataset(X, y)

    # Initialize model and score function
    model = build_regression_model("NonLinearNet")(
        input_dim=X.shape[1],
        output_dim=2,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(device)

    score_function = CQR()

    # Run experiment
    results = run_aci_experiment(
        model=model,
        score_function=score_function,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        alpha=alpha,
        epochs=epochs,
        gamma=gamma,
        verbose=True
    )

    # Print detailed results
    print("\nFinal Results:")
    print("-" * 60)
    for metric, value in results.items():
        if isinstance(value, (float, np.float32, np.float64)):
            print(f"{metric:<30} {value:.4f}")
        else:
            print(f"{metric:<30} {value}")


if __name__ == "__main__":
    main()
