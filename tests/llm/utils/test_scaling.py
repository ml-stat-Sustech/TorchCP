import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchcp.llm.utils.scaling import  LogisticRegression, BinningScaler, PlattBinningScaler, RecurrentScaler, BaseScaler

def test_base_scaler_fit():
    scaler = BaseScaler()
    X = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    scaler.fit(X, y)
    # Since fit does nothing, we just ensure it runs without error
    assert True

def test_base_scaler_predict():
    scaler = BaseScaler()
    X = torch.tensor([1.0, 2.0, 3.0])
    predictions = scaler.predict(X)
    assert torch.equal(predictions, X)
    
    
def test_logistic_regression():
    model = LogisticRegression()
    X = torch.randn(100, 1)
    y = torch.randint(0, 2, (100,1)).float()
    # Test predict
    pred = model(X)
    assert pred.shape == y.shape
    assert torch.all((pred >= 0) & (pred <= 1))


def test_binning_scaler():
    scaler = BinningScaler(n_bins=5)
    X = torch.randn(100)
    y = torch.randint(0, 2, (100,)).float()
    
    scaler.fit(X, y)
    assert hasattr(scaler, 'bins')
    assert hasattr(scaler, 'bin_prob')
    
    pred = scaler.predict(X)
    assert pred.shape == X.shape
    assert torch.all((pred >= 0) & (pred <= 1))

def test_binning_scaler_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins must be positive"):
        BinningScaler(n_bins=0)
        
def test_platt_binning_scaler():
    scaler = PlattBinningScaler(n_bins=5)
    X = torch.randn(100)
    y = torch.randint(0, 2, (100,)).float()
    
    scaler.fit(X, y)
    assert hasattr(scaler, 'platt')
    assert hasattr(scaler, 'binning')
    
    pred = scaler.predict(X)
    assert pred.shape == X.shape
    assert torch.all((pred >= 0) & (pred <= 1))


def test_recurrent_scaler():
    scaler = RecurrentScaler(hidden_size=32, num_layers=2)
    X = torch.randn(100, 10)  # Sequential data
    y = torch.randint(0, 2, (100,10)).float()
    
    # Test forward pass
    pred = scaler(X)
    assert pred.shape == y.shape
    
def test_edge_cases():
    # Test empty input
    with pytest.raises(RuntimeError):
        X = torch.tensor([])
        y = torch.tensor([])
        BinningScaler().fit(X, y)
    
    # Test single sample
    X = torch.tensor([0.5])
    y = torch.tensor([1.0])
    for Scaler in [BinningScaler, PlattBinningScaler]:
        scaler = Scaler()
        scaler.fit(X, y)
        pred = scaler.predict(X)
        assert pred.shape == X.shape

def test_numerical_stability():
    X = torch.tensor([1e-10, 0.5, 1-1e-10])
    y = torch.tensor([0.0, 0.5, 1.0])
    
    for Scaler in [BinningScaler, PlattBinningScaler]:
        scaler = Scaler()
        scaler.fit(X, y)
        pred = scaler.predict(X)
        assert torch.all(torch.isfinite(pred))
        assert torch.all((pred >= 0) & (pred <= 1))
        

def test_recurrent_scaler_fit():
    input_size = 10
    hidden_size = 20
    num_layers = 2
    model = RecurrentScaler(input_size, hidden_size, num_layers)
    X = torch.randn(100, 10) 
    y = torch.randint(0,2, (100,10), dtype=torch.float32)  
    model.fit(X,y)
    assert model.training == True 
    
    X = torch.randn(20, 10) 
    model.predict(X)
    assert model.training == False 

if __name__ == "__main__":
    pytest.main(["-v"])