import torch
import numpy as np
import random


__all__ = ["fix_randomness"]

def fix_randomness(seed=0):
    ### Fix randomness 
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    
