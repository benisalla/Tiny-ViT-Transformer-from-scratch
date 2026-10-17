import torch
import torch.nn as nn
import math

class ViTGELUActFun(nn.Module):
    """
    Implements the Gaussian Error Linear Unit (GELU) activation function. GELU is commonly
    used in Transformer models and is designed to provide smoother gating behavior than
    traditional nonlinearities like ReLU, by weighing inputs based on their magnitude.

    The GELU activation function approximates:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This approximation is non-monotonic and captures higher order moments of the input data,
    providing a probabilistic gate that smoothly varies between 0 and x.

    Methods:
        forward (x): Applies the GELU activation function to the input tensor.
    """
    def __init__(self):
        """
        Initializes the ViTGELUActFun module. This module does not contain any learnable
        parameters or states.
        """
        super(ViTGELUActFun, self).__init__()

    def forward(self, x):
        """
        Applies the GELU activation function to the input tensor using an explicit formula
        based on the original Gaussian Error Linear Unit formulation.

        Args:
            x (Tensor): The input tensor to which the GELU activation function will be applied.

        Returns:
            Tensor: The result tensor where the GELU activation function has been applied
                    element-wise to the input tensor.
        """
        # Apply the GELU activation function as per the defined formula
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
