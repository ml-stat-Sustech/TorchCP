import torch
import torch.nn as nn


class OrdinalClassifier(nn.Module):
    def __init__(self, classifier, phi = "abs", varphi = "abs"):
        super().__init__()
        self.classifier = classifier
        self.phi = phi
        self.varphi = varphi
        
        if self.phi == "abs":
            self.phi_function = torch.abs
        elif self.phi == "square":
            self.phi_function = torch.square
        else:
            raise NotImplementedError
            
            
        if self.varphi == "abs":
            self.varphi_function = lambda x: -torch.abs(x)
        elif self.phi == "square":
            self.varphi_function = lambda x: -torch.square(x)
        else:
            raise NotImplementedError
        

        
    def forward(self,x):
        assert x.shape[1] <= 2, "The input dimension must be greater than 2."
            
        x = self.classifier(x)
        
        # a cumulative summation
        x = torch.cat((x[:, :1], self.phi_function(x[:, 1:])), dim=1)

        x = torch.cumsum(x, dim=1)
        
        # the unimodal distribution
        x = self.varphi_function(x)
        return x