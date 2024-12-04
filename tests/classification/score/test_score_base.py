import pytest
import torch
from torchcp.classification.score.base import BaseScore


        
def test_base_call_method():
    """Test that BaseScore.__call__ raises NotImplementedError"""
    class MinimalBase(BaseScore):
        # Directly use BaseScore.__call__ without override
        def __call__(self, logits, labels=None):
            return super().__call__(logits, labels)
    
    scorer = MinimalBase()
    with pytest.raises(NotImplementedError):
        scorer(torch.tensor([[0.1, 0.9]]), torch.tensor([1]))
        



if __name__ == "__main__":
    pytest.main(["-v", "--cov=base", "--cov-report=term-missing"])