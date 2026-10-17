import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ViTSelfAttention(nn.Module):
    """
    The ViTSelfAttention class implements the self-attention mechanism used in Vision Transformer models.

    Attributes:
        n_embd (int): Dimensionality of the embeddings.
        h_size (int): Size of each attention head.
        n_head (int): Number of attention heads.
        d_rate (float): Dropout rate applied to the attention scores.
        n_patch (int): Number of patches into which the input image is divided.
        c_attn (nn.Linear): Linear layer to compute query, key, and value matrices.
        c_proj (nn.Linear): Linear layer to project the output of the attention mechanism.
        attn_dropout (nn.Dropout): Dropout layer applied to the attention scores.
        resid_dropout (nn.Dropout): Dropout layer applied after the final projection.
        flash (bool): Flag indicating the availability of the torch's native 'scaled_dot_product_attention'.

    Methods:
        forward (x, training=True): Defines the forward pass of the ViTSelfAttention layer.
    """
    def __init__(self, n_patch, n_embd, h_size, d_rate=0.0, bias=False):
        """
        Initializes the ViTSelfAttention module with the specified attributes.

        Args:
            n_patch (int): Number of patches into which the input image is divided.
            n_embd (int): Dimensionality of the embeddings.
            h_size (int): Size of each attention head.
            d_rate (float): Dropout rate.
            bias (bool): Whether to add bias in linear transformations.
        """
        super(ViTSelfAttention, self).__init__()
        assert n_embd % h_size == 0, "Embedding dimension must be divisible by the size of the attention head."

        self.n_embd = n_embd
        self.h_size = h_size
        self.n_head = n_embd // h_size
        self.d_rate = d_rate
        self.n_patch = n_patch

        # Layers for transforming inputs to queries, keys, and values
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Dropout layers for attention and output
        self.attn_dropout = nn.Dropout(d_rate)
        self.resid_dropout = nn.Dropout(d_rate)

        # Check if PyTorch has native support for scaled dot-product attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # Pre-computed attention bias for causal masking (if not using native attention)
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(n_patch, n_patch)).view(1, 1, n_patch, n_patch))

    def forward(self, x, training=True):
        """
        Defines the forward pass for the self-attention layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_patches, embedding_dim).
            training (bool): If true, applies dropout, otherwise bypasses dropout layers.

        Returns:
            Tensor: The output tensor after processing through the self-attention mechanism.
        """
        
        B, N, D = x.size()
        self.training = training

        # (B, N, 3*D) ==> 3 * (B, N, D)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, N, D) ==> (B, N, nh, hs) ==> (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)


        # (B, nh, N, hs) x (B, nh, hs, N) -> (B, nh, N, N)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.d_rate if self.training else 0, is_causal=False)
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v # (B, nh, N, N)  x  (B, nh, N, hs)   ==>   (B, nh, N, hs)

        # (B, nh, N, hs)  ==>  (B, nh, N, hs)  ==>  (B, N, D)
        y = y.transpose(1, 2).contiguous().view(B, N, D)

        return self.resid_dropout(self.c_proj(y))
