import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from transformers import set_seed

from examples.utils import build_reg_data
from torchcp.regression.loss import QuantileLoss
from torchcp.regression.predictor import EnsemblePredictor
from torchcp.regression.score import ABS, CQR
from torchcp.regression.utils import build_regression_model


def prepare_ensemble_dataset(X, y, train_ratio=0.5, batch_size=100):
    """
    Prepare datasets for ensemble methods.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        train_ratio (float): Ratio of training data
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: Training and test data loaders
    """
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


def run_ensemble_experiment(
        name,
        model,
        score_function,
        criterion,
        train_loader,
        test_loader,
        device,
        alpha=0.1,
        epochs=20,
        lr=0.01,
        ensemble_num=5,
        subset_num=500,
        verbose=True
):
    """
    Run ensemble prediction experiment.
    
    Args:
        name (str): Name of the experiment
        model (nn.Module): Neural network model
        score_function: Conformal score function
        criterion: Loss function
        train_loader: Training data loader
        test_loader: Test data loader
        device: Computing device
        alpha (float): Significance level
        epochs (int): Number of training epochs
        lr (float): Learning rate
        ensemble_num (int): Number of ensemble models
        subset_num (int): Number of samples in each subset
        verbose (bool): Whether to print detailed results
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'=' * 20} {name} {'=' * 20}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    aggregation_function = torch.mean

    predictor = EnsemblePredictor(score_function, model, aggregation_function)

    predictor.train(
        train_dataloader=train_loader,
        ensemble_num=ensemble_num,
        subset_num=subset_num,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer
    )

    results = predictor.evaluate(test_loader, alpha, verbose=verbose)
    print(f"Results: {results}")

    return results


def main():
    """Main function to run ensemble experiments."""
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed=1)

    # Hyperparameters
    epochs = 20
    alpha = 0.1
    hidden_dim = 64
    dropout = 0.5
    ensemble_num = 5
    subset_num = 500

    print("Starting ensemble prediction experiments...")

    # Load and prepare data
    X, y = build_reg_data(data_name="synthetic")
    train_loader, test_loader = prepare_ensemble_dataset(X, y)

    # Define experiments
    experiments = [
        {
            "name": "EnbPI (Sequential Distribution-free Ensemble)",
            "model": lambda: build_regression_model("NonLinearNet")(
                X.shape[1], 1, hidden_dim, dropout
            ),
            "score_function": ABS(),
            "criterion": nn.MSELoss(),
        },
        {
            "name": "EnCQR (Ensemble Conformal Quantile Regression)",
            "model": lambda: build_regression_model("NonLinearNet")(
                X.shape[1], 2, hidden_dim, dropout
            ),
            "score_function": CQR(),
            "criterion": QuantileLoss([alpha / 2, 1 - alpha / 2]),
        }
    ]

    # Run experiments and collect results
    results = {}
    for exp in experiments:
        model = exp["model"]().to(device)
        results[exp["name"]] = run_ensemble_experiment(
            name=exp["name"],
            model=model,
            score_function=exp["score_function"],
            criterion=exp["criterion"],
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            alpha=alpha,
            epochs=epochs,
            ensemble_num=ensemble_num,
            subset_num=subset_num,
            verbose=True
        )

    # Print comparative results
    print("\nComparative Results:")
    print("-" * 80)
    print(f"{'Method':<40} {'Coverage Rate':<20} {'Average Width':<20}")
    print("-" * 80)
    for method, result in results.items():
        print(f"{method:<40} {result['Coverage_rate']:.4f}{'':<16} "
              f"{result['Average_size']:.4f}")


if __name__ == "__main__":
    main()
