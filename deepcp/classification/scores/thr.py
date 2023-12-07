

from typing import Any


class THR:
    def __init__(self) -> None:
        pass
    
    
    def __call__(self, x, y):
        return 1-x[y]
    
    def predict(self,x):
        return 1-x