

import torch.nn as nn 
import torch
import torch.optim

from deepcp.utils.registry import Registry

ConfCalibrator_REGISTRY = Registry("ConfCalibrator")
ConfOptimizer_REGISTRY = Registry("ConfOptimizer")

class ConfCalibrator:
    @classmethod
    def registry_ConfCalibrator(cls, conf_calibrator):
        if conf_calibrator not in ConfCalibrator_REGISTRY.registered_names():
            raise NameError(f"The ConfCalibrator: {conf_calibrator} is not defined in DeepCP.")
        return ConfCalibrator_REGISTRY.get(conf_calibrator)
    
    @classmethod
    def registry_ConfOptimizer(cls, ConfOptimizer):
        if ConfOptimizer not in ConfOptimizer_REGISTRY.registered_names():
            raise NameError(f"The ConfOptimizer: {ConfOptimizer} is not defined in DeepCP.")
        return ConfOptimizer_REGISTRY.get(ConfOptimizer)
    
@ConfCalibrator_REGISTRY.register()
class Identity(nn.Module):
    def forward(self,batch_logits):
        return batch_logits
    
@ConfCalibrator_REGISTRY.register()
class TS(nn.Module):
    """Temperature Scaling"""
    def __init__(self,temperature=1) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self,batch_logits):
        return batch_logits/self.temperature
    
@ConfCalibrator_REGISTRY.register()
class oTS(nn.Module):
    """Optimal Temperature Scaling"""
    def __init__(self,temperature=1) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self,batch_logits):
        return batch_logits/self.temperature
    
    
@ConfOptimizer_REGISTRY.register()
def optimze_oTS( transformation, dataloader, device):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
       
        transformation.to(device)
        max_iters=10
        lr=0.01
        epsilon=0.01
        nll_criterion = nn.CrossEntropyLoss().to(device)
        T = transformation.temperature

        optimizer = torch.optim.SGD([transformation.temperature], lr=lr)
        for iter in range(max_iters):
            T_old = T.item()
            # print(T_old)
            for x, targets in dataloader:
                optimizer.zero_grad()
                x = x.to(device)
                x.requires_grad = True
                out = x/transformation.temperature
                loss = nll_criterion(out, targets.long().cuda())
                
                loss.backward()
                optimizer.step()
            T = transformation.temperature
            if abs(T_old - T.item()) < epsilon:
                break

        return transformation