import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, ConcatDataset
from transformers import set_seed

from torchcp.regression.loss import QuantileLoss, R2ccpLoss
from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.score import ABS, CQR, CQRR, CQRM, CQRFM, R2CCP
from torchcp.regression.utils import calculate_midpoints, build_regression_model
from examples.utils import build_reg_data


def prepare_dataset(X, y, train_ratio=0.4, cal_ratio=0.2, batch_size=100):
    """
    Prepare training, calibration and test datasets.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        train_ratio (float): Ratio of training data
        cal_ratio (float): Ratio of calibration data
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: Train, calibration and test data loaders
    """
    # Split indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index1 = int(len(indices) * train_ratio)
    split_index2 = int(len(indices) * (train_ratio + cal_ratio))
    part1, part2, part3 = np.split(indices, [split_index1, split_index2])
    
    # Scale features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[part1, :])
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[part1, :])), 
        torch.from_numpy(y[part1])
    )
    cal_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[part2, :])), 
        torch.from_numpy(y[part2])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[part3, :])), 
        torch.from_numpy(y[part3])
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    return train_loader, cal_loader, test_loader, cal_dataset


def run_experiment(name, model, score_function, criterion, train_loader, cal_loader, 
                  test_loader, device, epochs=20, lr=0.01, alpha=0.1, **kwargs):
    """
    Run a single conformal prediction experiment.
    
    Args:
        name (str): Name of the experiment
        model (nn.Module): Neural network model
        score_function: Conformal score function
        criterion: Loss function
        train_loader: Training data loader
        cal_loader: Calibration data loader
        test_loader: Test data loader
        device: Computing device
        epochs (int): Number of training epochs
        lr (float): Learning rate
        alpha (float): Significance level
        **kwargs: Additional arguments for specific methods
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*20} {name} {'='*20}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    predictor = SplitPredictor(score_function=score_function, model=model)
    
    if hasattr(criterion, 'midpoints'):
        predictor.train(
            train_dataloader=train_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer
        )
    else:
        predictor.train(
            train_dataloader=train_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            **kwargs
        )
    
    predictor.calibrate(cal_loader, alpha)
    results = predictor.evaluate(test_loader)
    print(f"Results: {results}")
    return results


def main():
    """Main function to run regression conformal prediction experiments."""
    print("Starting regression conformal prediction experiments...")
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed=1)
    alpha = 0.1
    epochs = 20
    input_dim = None  # Will be set after loading data
    
    # Load and prepare data
    X, y = build_reg_data(data_name="synthetic")
    input_dim = X.shape[1]
    train_loader, cal_loader, test_loader, cal_dataset = prepare_dataset(X, y)
    
    # Define experiments
    experiments = [
        {
            "name": "Split Conformal Prediction",
            "model": lambda: build_regression_model("NonLinearNet")(input_dim, 1, 64, 0.5),
            "score_function": ABS(),
            "criterion": nn.MSELoss(),
        },
        {
            "name": "Conformal Quantile Regression",
            "model": lambda: build_regression_model("NonLinearNet")(input_dim, 2, 64, 0.5),
            "score_function": CQR(),
            "criterion": QuantileLoss([alpha/2, 1-alpha/2]),
        },
        {
            "name": "CQRR",
            "model": lambda: build_regression_model("NonLinearNet")(input_dim, 2, 64, 0.5),
            "score_function": CQRR(),
            "criterion": QuantileLoss([alpha/2, 1-alpha/2]),
        },
        {
            "name": "CQRM",
            "model": lambda: build_regression_model("NonLinearNet")(input_dim, 3, 64, 0.5),
            "score_function": CQRM(),
            "criterion": QuantileLoss([alpha/2, 0.5, 1-alpha/2]),
        },
        {
            "name": "CQRFM",
            "model": lambda: build_regression_model("NonLinearNet")(input_dim, 3, 64, 0.5),
            "score_function": CQRFM(),
            "criterion": QuantileLoss([alpha/2, 0.5, 1-alpha/2]),
        },
    ]
    
    # Run experiments and collect results
    results = {}
    for exp in experiments:
        model = exp["model"]().to(device)
        results[exp["name"]] = run_experiment(
            name=exp["name"],
            model=model,
            score_function=exp["score_function"],
            criterion=exp["criterion"],
            train_loader=train_loader,
            cal_loader=cal_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            alpha=alpha
        )
    
    # Additional R2CCP experiment with special setup
    print("\nRunning R2CCP experiment...")
    K = 50
    train_and_cal_dataset = ConcatDataset([train_loader.dataset, cal_loader.dataset])
    train_and_cal_loader = torch.utils.data.DataLoader(
        train_and_cal_dataset, 
        batch_size=100, 
        shuffle=True,
        pin_memory=True
    )
    
    midpoints = calculate_midpoints(train_and_cal_loader, K).to(device)
    model = build_regression_model("NonLinearNet_with_Softmax")(
        input_dim, K, 1000, 0
    ).to(device)
    
    results["R2CCP"] = run_experiment(
        name="R2CCP",
        model=model,
        score_function=R2CCP(midpoints),
        criterion=R2ccpLoss(0.5, 0.2, midpoints),
        train_loader=train_loader,
        cal_loader=cal_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        alpha=alpha,
        lr=1e-4
    )
    
    # Print comparative results
    print("\nComparative Results:")
    print("-" * 80)
    print(f"{'Method':<30} {'Coverage Rate':<20} {'Average Width':<20}")
    print("-" * 80)
    for method, result in results.items():
        print(f"{method:<30} {result['Coverage_rate']:.4f}{'':<16} "
              f"{result['Average_size']:.4f}")


if __name__ == "__main__":
    main()