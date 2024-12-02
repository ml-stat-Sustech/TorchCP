import pytest
import torch
from torchcp.llm.predictors.conformal_llm  import (
    StoppingCriteriaSub, 
    NAME_TO_SCORE, 
    NAME_TO_SCALER, 
    ConformalLM, 
    DEFAULT_EPSILONS
)
def test_stopping_criteria_init():
    # Test default initialization
    criteria = StoppingCriteriaSub()
    assert criteria.input_length == 0
    assert criteria.stop_ids is None
    
    # Test custom initialization
    stop_ids = [50256, 50257]
    criteria = StoppingCriteriaSub(input_length=10, stop_ids=stop_ids)
    assert criteria.input_length == 10
    assert criteria.stop_ids == stop_ids

def test_stopping_criteria_call():
    criteria = StoppingCriteriaSub(input_length=2, stop_ids=[50256])
    
    # No stop tokens
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert not criteria(input_ids, None)
    
    # Stop token in one sequence
    input_ids = torch.tensor([[1, 2, 50256, 4], [5, 6, 7, 8]])
    assert not criteria(input_ids, None)
    
    # Stop token in all sequences
    input_ids = torch.tensor([[1, 2, 50256, 4], [5, 6, 50256, 8]])
    assert criteria(input_ids, None)
    
    # Multiple stop tokens
    criteria = StoppingCriteriaSub(input_length=2, stop_ids=[50256, 50257])
    input_ids = torch.tensor([[1, 2, 50256, 4], [5, 6, 50257, 8]])
    assert criteria(input_ids, None)

def test_name_to_score_mapping():
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    mask = torch.ones_like(X)
    
    # Test each scoring function
    for name, func in NAME_TO_SCORE.items():
        result = func(X, mask)
        assert isinstance(result, torch.Tensor)
        if name != 'none':
            assert result.shape == X.shape

def test_name_to_scaler_mapping():
    # Test each scaler initialization
    X = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,)).float()
    
    for name, scaler_class in NAME_TO_SCALER.items():
        if name == 'rnn':
            scaler = scaler_class(hidden_size=32, num_layers=2)
        else:
            scaler = scaler_class()
        assert hasattr(scaler, 'fit')
        assert hasattr(scaler, 'predict')

def test_conformal_lm_init():
    # Test default initialization
    clf = ConformalLM()
    assert clf.epsilons.equal(DEFAULT_EPSILONS)
    assert clf.scaling_type == "none"
    assert clf.set_score_function == NAME_TO_SCORE["none"]
    assert not clf.rejection
    assert clf.seed == 2024
    
    # Test custom initialization
    custom_epsilons = torch.linspace(0, 0.5, 10)
    clf = ConformalLM(
        epsilons=custom_epsilons,
        scaling_type="platt",
        scale_kwargs={"max_iter": 100},
        set_score_function_name="geo",
        rejection=True,
        seed=42
    )
    assert clf.epsilons.equal(custom_epsilons)
    assert clf.scaling_type == "platt"
    assert clf.scale_kwargs == {"max_iter": 100}
    assert clf.set_score_function == NAME_TO_SCORE["geo"]
    assert clf.rejection
    assert clf.seed == 42

def test_conformal_lm_validation():
    # Test empty epsilons
    with pytest.raises(ValueError, match="epsilons must be non-empty"):
        ConformalLM(epsilons=torch.tensor([]))
    
    # Test None epsilons
    with pytest.raises(ValueError, match="epsilons must be non-empty"):
        ConformalLM(epsilons=None)
    
    # Test invalid scaling_type
    with pytest.raises(ValueError, match="Invalid scaling_type:.*"):
        ConformalLM(scaling_type="invalid")
    
    # Test invalid score function
    with pytest.raises(ValueError, match="Invalid set_score_function_name:.*"):
        ConformalLM(set_score_function_name="invalid")

def test_default_epsilons():
    assert isinstance(DEFAULT_EPSILONS, torch.Tensor)
    assert len(DEFAULT_EPSILONS) == 101
    assert DEFAULT_EPSILONS[0] == 0
    assert DEFAULT_EPSILONS[-1] == 1
    assert torch.all(torch.diff(DEFAULT_EPSILONS) > 0)


if __name__ == "__main__":
    pytest.main(["-v"])