import torch.nn as nn
import torch
from einops import repeat
from einops.layers.torch import Rearrange

class ViTConv2dEmbedding(nn.Module):
    """
    Converts input images into a sequence of flattened 2D patches and applies a learned linear transformation to each patch. 
    Additionally, it prepends a learnable class token to the sequence of embedded patches.

    Attributes:
        p_size (int): The size of each patch (both height and width).
        ite (nn.Sequential): Sequential module containing a Conv2D layer for patch creation and flattening.
        cls_token (nn.Parameter): Learnable tensor representing the class token.

    Methods:
        forward(x): Takes an input tensor of images and returns their corresponding sequence of embeddings.
    """
    def __init__(self, n_embd, p_size, c_dim):
        super(ViTConv2dEmbedding, self).__init__()
        self.p_size = p_size
        self.ite = nn.Sequential(
            nn.Conv2d(c_dim, n_embd, kernel_size=p_size, stride=p_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        # (B, C, H, W)
        b, _, _, _ = x.shape
        x = self.ite(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class ViTLNEmbedding(nn.Module):
    """
    Applies a layer normalization to the flattened input patches before projecting them into an embedding space.
    This module also appends a learnable class token at the beginning of the sequence.

    Attributes:
        p_size (int): The size of each patch.
        c_dim (int): Number of input channels.
        ite (nn.Sequential): Sequential module containing LayerNorm and a Linear transformation.
        cls_token (nn.Parameter): Learnable tensor representing the class token.

    Methods:
        forward(x): Processes the input images into a sequence of embeddings with a class token.
    """
    def __init__(self, n_embd, p_size, c_dim):
        super(ViTLNEmbedding, self).__init__()
        self.p_size = p_size
        self.c_dim = c_dim
        self.ite = nn.Sequential(
            nn.LayerNorm(p_size * p_size * c_dim),
            nn.Linear(p_size * p_size * c_dim, n_embd),
            nn.LayerNorm(n_embd),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        # (B, C, H, W)
        B, _, _, _ = x.shape
        P, C = self.p_size, self.c_dim
        # (B, C, H, W) => (B, H//P, W//P, P, P, C) => (B, (H//P) * (W//P), P*P*C)
        x = (x.unfold(2, P, P)
            .unfold(3, P, P)
            .permute(0, 2, 3, 4, 5, 1)
            .contiguous()
            .reshape(B, -1, P * P * C))
        x = self.ite(x)
        cls_toks = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_toks, x], dim=1)
        return x

    
class ViTPyCon2DEmbedding(nn.Module):
    """
    This class performs a similar role to ViTConv2dEmbedding but uses different tensor reshaping strategies. 
    It transforms input images into a sequence of embedded patches and prepends a learnable class token.

    Attributes:
        n_embd (int): Dimensionality of the embedding space.
        p_size (int): The size of each patch.
        c_dim (int): Number of input channels.
        ite (nn.Conv2d): Convolutional layer to create patch embeddings directly.
        cls_token (nn.Parameter): Learnable tensor representing the class token.

    Methods:
        forward(x): Processes input images into a sequence of embeddings with a class token.
    """
    def __init__(self, n_embd, p_size, c_dim):
        super(ViTPyCon2DEmbedding, self).__init__()
        self.n_embd = n_embd
        self.p_size = p_size
        self.c_dim = c_dim
        self.ite = nn.Conv2d(c_dim, n_embd, kernel_size=p_size, stride=p_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        # (B, C, H, W)
        B, _, _, _ = x.shape
        x = self.ite(x)
        # or Rearrange('b e (h) (w) -> b (h w) e')
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, self.n_embd)
        # or repeat(self.cls_token, '() n e -> b n e', b=B)
        cls_toks = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_toks, x], dim=1)
        return x