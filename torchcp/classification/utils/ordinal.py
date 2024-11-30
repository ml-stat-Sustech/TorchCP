import torch
import torch.nn as nn


class OrdinalClassifier(nn.Module):
    """
    Method: Ordinal Classifier 
    Paper: Conformal Prediction Sets for Ordinal Classification
    Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html

    Args:
        classifier (nn.Module): A PyTorch classifier model.
        phi (str, optional): The transformation function for the classifier output. Default is "abs". Options are "abs" and "square".
        varphi (str, optional): The transformation function for the cumulative sum. Default is "abs". Options are "abs" and "square".

    Attributes:
        classifier (nn.Module): The classifier model.
        phi (str): The transformation function for the classifier output.
        varphi (str): The transformation function for the cumulative sum.
        phi_function (callable): The transformation function applied to the classifier output.
        varphi_function (callable): The transformation function applied to the cumulative sum.

    Methods:
        forward(x):
            Forward pass of the model.
    
    Examples::
        >>> classifier = nn.Linear(10, 5)
        >>> model = OrdinalClassifier(classifier, phi="abs", varphi="abs")
        >>> x = torch.randn(3, 10)
        >>> output = model(x)
        >>> print(output)
    """
    
    def __init__(self, classifier, phi="abs", varphi="abs"):
        super().__init__()
        self.classifier = classifier
        self.phi = phi
        self.varphi = varphi

        phi_options = {"abs": torch.abs, "square": torch.square}
        varphi_options = {"abs": lambda x: -torch.abs(x), "square": lambda x: -torch.square(x)}

        if phi not in phi_options:
            raise NotImplementedError(f"phi function '{self.phi}' is not implemented. Options are 'abs' and 'square'.")
        if varphi not in varphi_options:
            raise NotImplementedError(f"varphi function '{self.varphi}' is not implemented. Options are 'abs' and 'square'.")

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the classifier, phi_function, and varphi_function.
        """
        if x.shape[1] <= 2:
            raise ValueError("The input dimension must be greater than 2.")

        x = self.classifier(x)

        # a cumulative summation
        x = torch.cat((x[:, :1], self.phi_function(x[:, 1:])), dim=1)

        x = torch.cumsum(x, dim=1)

        # the unimodal distribution
        x = self.varphi_function(x)
        return x
