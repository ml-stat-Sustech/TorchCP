import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from torchcp.classification.trainer.ts_trainer import TSTrainer

class SimpleModel(nn.Module):
    """A simple neural network model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
        
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def setup_data():
    """Fixture to create synthetic data for testing"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    num_samples = 100
    input_dim = 10
    num_classes = 3
    
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return loader

@pytest.fixture
def setup_model():
    """Fixture to create and pre-train a model for testing"""
    model = SimpleModel()
    # Pre-train the model with some random data
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    return model

def test_ts_trainer_initialization(setup_model):
    """Test the initialization of TSTrainer"""
    model = setup_model
    trainer = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu"),
        verbose=False
    )
    
    assert isinstance(trainer.model.temperature, torch.nn.Parameter)
    assert trainer.model.temperature.item() == 1.0
    assert trainer.device == torch.device("cpu")
    
    trainer = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu"),
        verbose=True
    )

def test_ts_trainer_training(setup_model, setup_data):
    """Test the temperature parameter training process"""
    model = setup_model
    train_loader = setup_data
    
    trainer = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu"),
        verbose=False
    )
    
    initial_temperature = trainer.model.temperature.item()
    
    trainer.train(
        train_loader=train_loader,
        lr=0.01,
        num_epochs=50
    )
    
    final_temperature = trainer.model.temperature.item()
    
    # Ensure temperature parameter has been updated
    assert final_temperature != initial_temperature
    # Ensure temperature parameter is positive
    assert final_temperature > 0

def test_calibrated_predictions(setup_model, setup_data):
    """Test the calibrated prediction results"""
    model = setup_model
    train_loader = setup_data
    
    trainer = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu"),
        verbose=False
    )
    
    trainer.train(
        train_loader=train_loader,
        lr=0.01,
        num_epochs=50
    )
    
    # Test predictions on a single batch
    inputs, _ = next(iter(train_loader))
    
    # Get predictions from both original and calibrated models
    with torch.no_grad():
        original_outputs = torch.softmax(model(inputs), dim=1)
        calibrated_outputs = torch.softmax(trainer.model(inputs), dim=1)
        
        original_max_probs, _ = torch.max(original_outputs, dim=1)
        calibrated_max_probs, _ = torch.max(calibrated_outputs, dim=1)
    
    # Check basic properties of predictions
    assert torch.all(calibrated_outputs >= 0)
    assert torch.all(calibrated_outputs <= 1)
    # Check if probabilities sum to 1
    assert torch.allclose(torch.sum(calibrated_outputs, dim=1), 
                         torch.ones(calibrated_outputs.shape[0]))

def test_error_handling(setup_model):
    """Test error handling scenarios"""
    model = setup_model
    
    # Test initialization with negative temperature
    with pytest.raises(Exception):
        TSTrainer(
            model=model,
            init_temperature=-1.0,
            device=torch.device("cpu")
        )
    
    # Test training with empty data loader
    trainer = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu")
    )
    
    empty_loader = DataLoader(TensorDataset(torch.tensor([]), torch.tensor([])))
    with pytest.raises(Exception):
        trainer.train(train_loader=empty_loader)

def test_device_handling(setup_model, setup_data):
    """Test device handling (CPU/GPU)"""
    model = setup_model
    train_loader = setup_data
    
    # Test CPU device
    trainer_cpu = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu"),
        verbose=False
    )
    
    assert trainer_cpu.device == torch.device("cpu")
    assert next(trainer_cpu.model.parameters()).device == torch.device("cpu")
    
    # Test GPU device if CUDA is available
    if torch.cuda.is_available():
        trainer_gpu = TSTrainer(
            model=model,
            init_temperature=1.0,
            device=torch.device("cuda"),
            verbose=False
        )
        
        assert str(trainer_gpu.device).startswith("cuda")
        assert str(next(trainer_gpu.model.parameters()).device).startswith("cuda")
        
        
def test_training_metrics(setup_model, setup_data):
    """Test the training metrics calculation and verbose output"""
    model = setup_model
    train_loader = setup_data
    
    # Test with verbose=True to check metrics output
    trainer = TSTrainer(
        model=model,
        init_temperature=1.0,
        device=torch.device("cpu"),
        verbose=True
    )
    
    # Capture the print output
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    trainer.train(
        train_loader=train_loader,
        lr=0.01,
        num_epochs=50
    )
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    # Check if metrics are printed
    assert "Before scaling - NLL:" in output
    assert "ECE:" in output
    assert "After scaling - NLL:" in output
    assert "Optimal temperature:" in output
    
    # Test metrics values
    for line in output.split('\n'):
        if "Before scaling" in line or "After scaling" in line:
            # Extract NLL and ECE values
            metrics = line.split('-')[1].strip()
            nll = float(metrics.split(',')[0].split(':')[1])
            ece = float(metrics.split(',')[1].split(':')[1])
            
            # Check if metrics are valid numbers
            assert isinstance(nll, float)
            assert isinstance(ece, float)
            assert 0 <= ece <= 1  # ECE should be between 0 and 1
            assert nll >= 0  # NLL should be non-negative

def test_ece_loss_calculation():
    """Test the Expected Calibration Error (ECE) loss calculation"""
    from torchcp.classification.trainer.ts_trainer import _ECELoss
    
    # Create ECE loss instance
    ece_criterion = _ECELoss(n_bins=15)
    
    # Create synthetic logits and labels
    batch_size = 100
    num_classes = 3
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Calculate ECE
    ece = ece_criterion(logits, labels)
    
    # Basic checks for ECE
    assert isinstance(ece, torch.Tensor)
    assert ece.dim() == 1
    assert ece.size(0) == 1
    assert 0 <= ece.item() <= 1  # ECE should be between 0 and 1
    
    # Test ECE with perfect predictions
    # Create logits that would give perfect predictions
    perfect_logits = torch.zeros(batch_size, num_classes)
    perfect_logits[range(batch_size), labels] = 10.0  # High confidence for correct class
    
    perfect_ece = ece_criterion(perfect_logits, labels)
    assert perfect_ece.item() < 0.1  # Should be close to 0 for perfect predictions
    
    # Test ECE with completely wrong predictions
    # Create logits that would give wrong predictions with high confidence
    wrong_labels = (labels + 1) % num_classes
    wrong_logits = torch.zeros(batch_size, num_classes)
    wrong_logits[range(batch_size), wrong_labels] = 10.0
    
    wrong_ece = ece_criterion(wrong_logits, labels)
    assert wrong_ece.item() > 0.5  # Should be high for completely wrong predictions
    
    # Test with different number of bins
    ece_criterion_more_bins = _ECELoss(n_bins=20)
    ece_more_bins = ece_criterion_more_bins(logits, labels)
    assert isinstance(ece_more_bins, torch.Tensor)
    assert 0 <= ece_more_bins.item() <= 1

def test_verbose_output_consistency(setup_model, setup_data):
    """Test the consistency of verbose output across multiple runs"""
    model = setup_model
    train_loader = setup_data
    
    # Run training multiple times with same parameters
    temperatures = []
    nlls_before = []
    nlls_after = []
    eces_before = []
    eces_after = []
    
    for _ in range(3):
        # Capture output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        trainer = TSTrainer(
            model=setup_model,  # Use fresh model each time
            init_temperature=1.0,
            device=torch.device("cpu"),
            verbose=True
        )
        
        trainer.train(
            train_loader=train_loader,
            lr=0.01,
            num_epochs=50
        )
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Extract metrics from output
        for line in output.split('\n'):
            if "Optimal temperature:" in line:
                temperatures.append(float(line.split(':')[1]))
            elif "Before scaling" in line:
                metrics = line.split('-')[1].strip()
                nlls_before.append(float(metrics.split(',')[0].split(':')[1]))
                eces_before.append(float(metrics.split(',')[1].split(':')[1]))
            elif "After scaling" in line:
                metrics = line.split('-')[1].strip()
                nlls_after.append(float(metrics.split(',')[0].split(':')[1]))
                eces_after.append(float(metrics.split(',')[1].split(':')[1]))
    
    # Check consistency of results
    temp_std = np.std(temperatures)
    nll_before_std = np.std(nlls_before)
    nll_after_std = np.std(nlls_after)
    ece_before_std = np.std(eces_before)
    ece_after_std = np.std(eces_after)
    
    # Results should be relatively consistent across runs
    assert temp_std < 0.1  # Temperature should be stable
    assert nll_before_std < 0.1  # NLL before scaling should be consistent
    assert nll_after_std < 0.1   # NLL after scaling should be consistent
    assert ece_before_std < 0.1  # ECE before scaling should be consistent
    assert ece_after_std < 0.1   # ECE after scaling should be consistent