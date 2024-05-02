import torch.nn as nn
from model.vit_feedforward import ViTFeedForwardMLP
from model.vit_layernorm import ViTLayerNorm
from model.vit_self_attention import ViTSelfAttention

class ViTBlock(nn.Module):
    """
    Represents a single block of a Vision Transformer model, consisting of a self-attention layer and a 
    position-wise feedforward network, each followed by layer normalization. The block follows the typical
    architecture of Transformer blocks used in NLP, adapted for vision tasks.

    Attributes:
        attn_ln (ViTLayerNorm): Layer normalization applied before the self-attention mechanism.
        attn (ViTSelfAttention): The self-attention mechanism of the block.
        ff_ln (ViTLayerNorm): Layer normalization applied before the feedforward network.
        ff_mlp (ViTFeedForwardMLP): The feedforward network that processes the output from the self-attention mechanism.

    Methods:
        forward(x): Defines the computation performed at every call of a ViTBlock, which includes the application of 
                    self-attention followed by a feedforward network, with residual connections around each.
    """
    def __init__(self, n_patch, n_embd, h_size, h_dim=None, d_rate=0.0, bias=False):
        """
        Initializes a Vision Transformer block with the necessary components and configurations.

        Args:
            n_patch (int): The number of patches into which the input images are divided.
            n_embd (int): The dimensionality of the input embeddings.
            h_size (int): The dimensionality of each head in the multi-head attention mechanism.
            h_dim (int, optional): The dimensionality of the inner layer of the feedforward network. If None,
            it defaults to 4 times `n_embd`.
            d_rate (float): Dropout rate applied within the attention mechanism and feedforward network.
            bias (bool): Whether to include bias in the layer normalizations and other components.

        Note:
            The feedforward network typically expands the embeddings inside its first layer to `h_dim`, which
            is often set to 4 times the size of `n_embd`, and then projects them back to `n_embd` dimensions.
        """
        super(ViTBlock, self).__init__()
        self.attn_ln = ViTLayerNorm(n_dim=n_embd, bias=bias)
        self.attn = ViTSelfAttention(n_patch=n_patch, n_embd=n_embd, h_size=h_size, d_rate=d_rate, bias=bias)
        self.ff_ln = ViTLayerNorm(n_dim=n_embd, bias=bias)
        self.ff_mlp = ViTFeedForwardMLP(n_embd=n_embd, h_dim=h_dim, d_rate=d_rate, bias=bias)

    def forward(self, x):
        """
        Applies the operations of a Vision Transformer block to the input tensor.

        Args:
            x (Tensor): The input tensor to the ViT block with shape (batch_size, num_patches, embedding_dim).

        Returns:
            Tensor: The output tensor of the ViT block after processing, having the same shape as the input.
        """
        
        x = x + self.attn(self.attn_ln(x))        
        x = x + self.ff_mlp(self.ff_ln(x))
        return x
