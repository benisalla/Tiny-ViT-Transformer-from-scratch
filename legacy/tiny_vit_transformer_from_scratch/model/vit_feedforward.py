import torch.nn as nn
from tiny_vit_transformer_from_scratch.model.vit_glu_act_fun import ViTGELUActFun

class ViTFeedForwardMLP(nn.Module):
    """
    Implements the feedforward multi-layer perceptron (MLP) component of a Vision Transformer block. This
    MLP consists of two linear transformations with a GELU activation function in-between, which is typical
    for Transformer architectures. The MLP is used to process features after attention mechanisms within
    Transformer blocks.

    Attributes:
        vit_mlp (nn.Sequential): A sequence of layers comprising the MLP. It includes two fully connected
        linear layers with a GELU activation and a dropout layer in between.

    Methods:
        forward (x): Defines the forward pass of the MLP.
    """
    def __init__(self, n_embd, h_dim=None, d_rate=0.0, bias=False):
        """
        Initializes the ViTFeedForwardMLP module with specified dimensions and configurations.

        Args:
            n_embd (int): Dimensionality of input features.
            h_dim (int, optional): Dimensionality of the hidden layer. If not provided, it defaults to
            four times the size of input features (common practice in Transformers).
            d_rate (float): Dropout rate applied after the activation function to prevent overfitting.
            bias (bool): Whether to include bias terms in the linear layers. Defaults to False.
        """
        super(ViTFeedForwardMLP, self).__init__()

        if not h_dim:
            h_dim = 6 * n_embd 

        # by mistake the model train with: h_dim = 4 * n_embd 
        # so to load pre-trained model i will use 4 * n_embd instead of h_dim
        # notice that h_dim=3072 however it is 2048
        self.vit_mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),  
            nn.GELU(),  # GELU activation function (can replace with ViTGELUActFun() if custom implementation is needed)
            nn.Linear(4 * n_embd, n_embd, bias=bias), 
            nn.Dropout(d_rate),  
        )

    def forward(self, x):
        """
        Forward pass for the ViTFeedForwardMLP. Applies a series of transformations defined in `vit_mlp` to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_patches, embedding_dim) representing the output from the previous
                        Transformer layer or input embedding.

        Returns:
            Tensor: Output tensor of the same shape as input, after processing through the MLP.
        """
        x = self.vit_mlp(x)  
        return x
