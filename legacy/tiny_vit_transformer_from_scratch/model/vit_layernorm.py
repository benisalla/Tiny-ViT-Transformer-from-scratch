import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTLayerNorm(nn.Module):
    """
    Implements a Layer Normalization module, which is a fundamental component in Transformer architectures,
    including Vision Transformers. Layer Normalization is applied across all features in each data sample 
    individually and helps in stabilizing the learning process.

    Attributes:
        w (nn.Parameter): Scale parameters (learnable) of the layer normalization.
        b (nn.Parameter, optional): Bias parameters (learnable) of the layer normalization. Included only if bias is True.
    
    Methods:
        forward (x): Applies layer normalization to the input tensor.
    """
    def __init__(self, n_dim, bias=False):
        """
        Initializes the ViTLayerNorm module with the necessary parameters and weights.

        Args:
            n_dim (int): The feature dimensionality of the input tensor to be normalized.
            bias (bool): If True, bias parameters are included in the layer normalization,
            and are initialized to zero.

        Note:
            The scale parameters are initialized to ones, ensuring that the initial state of
            the layer normalization is neutral, affecting neither the scale nor the shift
            of the input tensor.
        """
        super(ViTLayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(n_dim))  # Initialize scale parameters to ones
        self.b = nn.Parameter(torch.zeros(n_dim)) if bias else None  # Initialize bias parameters to zeros if bias is True

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (Tensor): The input tensor to be normalized. Expected to have dimensions [batch_size, ..., n_dim]
                        where `n_dim` is the dimension over which normalization is applied.

        Returns:
            Tensor: The normalized tensor, which has the same shape as the input tensor.
        """
        # Apply layer normalization using PyTorch's functional API. The shape of the scale and bias parameters
        # is automatically broadcasted to match the input dimensions as necessary.
        return F.layer_norm(x, self.w.shape, self.w, self.b, 1e-5)  # Epsilon of 1e-5 is used to improve numerical stability.
