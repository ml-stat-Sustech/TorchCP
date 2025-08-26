import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.regression_cqr_synthetic import prepare_dataset
from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.utils import build_regression_model

# Ensure these classes are correctly imported depending on your project setup:
# Option 1: If my_norabs.py is located in the same directory:
# from my_norabs import DifficultyEstimator, NorABS
# Option 2: If integrated into the torchcp library:
from torchcp.regression.score import NorABS, DifficultyEstimator


def extract_variance_as_function(x_batch, predicts_batch):
    """
    Custom difficulty function that replicates the behavior of the 'variance' mode.

    Args:
        x_batch (Tensor): Input features (unused here but kept for API consistency).
        predicts_batch (Tensor): Predictions with shape (batch_size, 2),
                                 where the second column is interpreted as variance.

    Returns:
        Tensor: A vector representing the difficulty score based on variance.
    """
    if predicts_batch is None or predicts_batch.ndim != 2 or predicts_batch.shape[-1] != 2:
        raise ValueError("This function requires `predicts_batch` to have shape (batch_size, 2).")
    return predicts_batch[:, 1]


def run_experiment(mode='variance'):
    """
    Executes a conformal prediction experiment using a lazily calibrated DifficultyEstimator.

    Args:
        mode (str): Difficulty estimation strategy.
                    Supported values: 'variance', 'knn_residual', 'knn_label', 'knn_distance', 'function'.
    """
    print(f"\n--- Running Experiment in '{mode}' mode ---")
    
    # 1. Prepare data loaders for training, calibration, and testing
    train_loader, cal_loader, test_loader = prepare_dataset(train_ratio=0.4, cal_ratio=0.2, batch_size=128)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Extract feature dimensionality from a sample batch to build the model
    sample_x, _ = next(iter(train_loader))
    feature_dim = sample_x.shape[1]

    # 2. Initialize the regression model (untrained at this stage)
    model = build_regression_model("GaussianRegressionModel")(feature_dim, 64, 0.5)
    
    # 3. Define the NorABS score function for difficulty-aware conformity estimation
    score_function = NorABS(data_loader=train_loader,
                            estimate_type=mode,
                            k=20,
                            scalar=True,
                            beta=0.01,
                            device=device,
                            custom_function=extract_variance_as_function)

    # 4. Wrap the score function and model inside a SplitPredictor instance
    predictor = SplitPredictor(score_function=score_function, model=model, alpha=0.1, device=device)

    # 5. Train the base regression model using the predictor’s utility function
    print("Training the base regression model...")
    predictor.train(train_loader, epochs=100, lr=0.01, device=device, verbose=True)
    # After training, the model is stored within `predictor.model`

    # 6. Calibrate the predictor on the calibration set.
    #    This step computes nonconformity scores and determines the quantile q_hat.
    print("Calibrating the predictor (computing q_hat)...")
    predictor.calibrate(cal_loader)

    # 7. Online evaluation on the test set (transductive protocol).
    print("Evaluating on the test set...")
    cover_count, total, set_size_sum = 0, 0, 0
    
    # Maintain a mutable calibration set that is updated sequentially
    online_cal_dataset = list(cal_loader.dataset)
    test_dataset = test_loader.dataset

    for i in tqdm(range(len(test_dataset)), desc="Online Evaluation"):
        x_test, y_test = test_dataset[i]
        x_test_device = x_test.to(device).unsqueeze(0)  # Predictor expects a batch dimension

        # Generate prediction intervals for the test sample
        prediction_intervals = predictor.predict(x_test_device)[0][0]
        
        # Evaluate coverage and interval size
        covered = (prediction_intervals[0] <= y_test <= prediction_intervals[1])
        cover_count += int(covered)
        set_size_sum += (prediction_intervals[1] - prediction_intervals[0]).item()
        total += 1
        
        # Update the calibration set with the new (x, y) pair
        online_cal_dataset.append((x_test, y_test))
        online_cal_loader = DataLoader(online_cal_dataset, batch_size=128, shuffle=True)
        
        # Recalibrate the predictor with the expanded calibration set
        predictor.calibrate(online_cal_loader)

    # Compute final evaluation metrics
    coverage_rate = cover_count / total
    average_set_size = set_size_sum / total
    alpha = predictor.alpha

    print(f"\nResults for '{mode}' mode:")
    print(f"Online CP Coverage Rate: {coverage_rate:.4f} (Target: {1-alpha:.2f})")
    print(f"Online CP Average Set Size: {average_set_size:.4f}")


if __name__ == "__main__":
    # Run the experiment under different difficulty estimation modes
    run_experiment(mode='variance')
    run_experiment(mode='knn_residual')
    run_experiment(mode='knn_label')
    run_experiment(mode='knn_distance')
    run_experiment(mode='function')
