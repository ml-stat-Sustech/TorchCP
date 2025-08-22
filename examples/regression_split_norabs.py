import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.regression_cqr_synthetic import prepare_dataset
from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.utils import build_regression_model

# Make sure these classes are importable from your project structure
# Option 1: If you created my_norabs.py in the same directory
# from my_norabs import DifficultyEstimator, NorABS
# Option 2: If you integrated them into the library
from torchcp.regression.score import NorABS, DifficultyEstimator


def extract_variance_as_function(x_batch, predicts_batch):
    """
    A custom difficulty function that mimics the 'variance' mode.
    """
    if predicts_batch is None or predicts_batch.ndim != 2 or predicts_batch.shape[-1] != 2:
        raise ValueError("This function requires `predicts_batch` to have shape (batch_size, 2).")
    return predicts_batch[:, 1]


def run_experiment(mode='variance'):
    """
    Runs a conformal prediction experiment with a lazily calibrated DifficultyEstimator.

    Args:
        mode (str): The difficulty estimation mode to use. 
                    Can be 'variance' or 'knn_residual'.
    """
    print(f"\n--- Running Experiment in '{mode}' mode ---")
    
    # 1. Get Dataloader
    train_loader, cal_loader, test_loader = prepare_dataset(train_ratio=0.4, cal_ratio=0.2, batch_size=128)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Extract a sample batch to get feature dimensions for model creation
    sample_x, _ = next(iter(train_loader))
    feature_dim = sample_x.shape[1]

    # 2. Build the appropriate regression model (but do not train it yet)
    model = build_regression_model("GaussianRegressionModel")(feature_dim, 64, 0.5)

    # 3. Create the lightweight DifficultyEstimator instance
    print("Initializing lightweight Difficulty Estimator...")
    diff_estimator = DifficultyEstimator(cal_loader=cal_loader, 
                                         estimator_type=mode,
                                         k=20,
                                         scalar=True,
                                         beta=0.01,
                                         device=device,
                                         custom_function=extract_variance_as_function)

    # 4. Create the NorABS score function and the main SplitPredictor
    score_function = NorABS(difficulty_estimator=diff_estimator)
    predictor = SplitPredictor(score_function=score_function, model=model, alpha=0.1, device=device)

    # 5. Train the model using the predictor's built-in helper function
    print("Training the base regression model...")
    predictor.train(train_loader, epochs=100, lr=0.01, device=device, verbose=True)
    # The trained model is now stored inside `predictor.model`

    # 6. Explicitly calibrate the DifficultyEstimator.
    #    This is the crucial step that uses the now-trained model to prepare the estimator.
    print("Calibrating the Difficulty Estimator...")
    diff_estimator.calibrate(predictor._model)
    
    # 7. Calibrate the predictor itself.
    #    This step computes the non-conformity scores (using the now-calibrated estimator)
    #    and finds the quantile (q_hat).
    print("Calibrating the predictor (computing q_hat)...")
    predictor.calibrate(cal_loader)

    # 8. Evaluate on the test set using an online/transductive evaluation protocol
    print("Evaluating on the test set...")
    cover_count = 0
    total = 0
    set_size_sum = 0
    
    # Create a mutable list for the online calibration set
    online_cal_dataset = list(cal_loader.dataset)
    test_dataset = test_loader.dataset

    for i in tqdm(range(len(test_dataset)), desc="Online Evaluation"):
        x_test, y_test = test_dataset[i]
        x_test_device = x_test.to(device).unsqueeze(0) # Predictor expects a batch

        # Prediction generates the interval
        prediction_intervals = predictor.predict(x_test_device)[0][0]
        
        # Check coverage and interval size
        covered = (prediction_intervals[0] <= y_test <= prediction_intervals[1])
        cover_count += int(covered)
        set_size_sum += (prediction_intervals[1] - prediction_intervals[0]).item()
        total += 1
        
        # Update the calibration set with the new test point for the next iteration
        online_cal_dataset.append((x_test, y_test))
        online_cal_loader = DataLoader(online_cal_dataset, batch_size=128, shuffle=True)
        
        # Re-calibrate the predictor with the updated dataset
        predictor.calibrate(online_cal_loader)

    coverage_rate = cover_count / total
    average_set_size = set_size_sum / total
    alpha = predictor.alpha

    print(f"\nResults for '{mode}' mode:")
    print(f"Online CP Coverage Rate: {coverage_rate:.4f} (Target: {1-alpha:.2f})")
    print(f"Online CP Average Set Size: {average_set_size:.4f}")

if __name__ == "__main__":
    # Run the experiment for different difficulty estimation methods
    run_experiment(mode='variance')
    run_experiment(mode='knn_residual')
    run_experiment(mode='knn_label')
    run_experiment(mode='knn_distance')
    run_experiment(mode='function')