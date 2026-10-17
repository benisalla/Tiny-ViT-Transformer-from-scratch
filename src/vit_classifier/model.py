"""A Vision Transformer (ViT) written from scratch, read top to bottom.

    image (B, 3, H, W)
      -> PatchEmbedding      cut into P x P patches, one d_model vector per patch   (B, N, D)
      -> prepend [CLS] token, add learned position embeddings                       (B, N+1, D)
      -> n_layers x Block    pre-norm: x + Attention(LN(x)), then x + MLP(LN(x))     (B, N+1, D)
      -> LayerNorm, take [CLS] (or the mean of the patch tokens)                    (B, D)
      -> Linear head                                                                (B, C_out)

B = batch, N = (H / P) * (W / P) patches, D = d_model, C_out = num_classes.
Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vit_classifier.config import ModelConfig

# ---------------------------------------------------------------------------
# 1. Patch embedding: image -> sequence of tokens
# ---------------------------------------------------------------------------


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, P*P*C): each row is one flattened patch, row-major order.

    Patch n covers rows (n // W') * P ... and columns (n % W') * P ..., W' = W / P.
    """
    B, C, H, W = images.shape
    P = patch_size
    # (B, C, H/P, P, W/P, P) -> (B, H/P, W/P, P, P, C) -> (B, N, P*P*C)
    x = images.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, (H // P) * (W // P), P * P * C)


class PatchEmbedding(nn.Module):
    """Turn an image into N tokens of width d_model.

    Two equivalent ways to do it (both are a linear map of each patch):
    - "linear": flatten each patch, then LayerNorm -> Linear -> LayerNorm.
      The extra LayerNorms (from the first version of this project) make
      training a little more stable when trained from scratch.
    - "conv": a Conv2d with kernel_size = stride = patch_size. Each output
      pixel of the convolution sees exactly one patch, so this is the same
      linear map, computed in one fused operation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.kind = config.patch_embed
        self.patch_size = config.patch_size
        patch_dim = config.patch_size**2 * config.in_channels
        if self.kind == "linear":
            self.proj = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, config.d_model),
                nn.LayerNorm(config.d_model),
            )
        else:
            self.proj = nn.Conv2d(
                config.in_channels, config.d_model, config.patch_size, stride=config.patch_size
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return self.proj(patchify(images, self.patch_size))  # (B, N, D)
        x = self.proj(images)  # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, D), same patch order as patchify


# ---------------------------------------------------------------------------
# 2. Multi-head self-attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Every token looks at every other token (no causal mask: images have no "future").

    Each head h computes softmax(Q_h K_h^T / sqrt(head_dim)) V_h; the heads are
    concatenated and mixed by a final linear layer.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)  # Q, K and V in one matmul
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D = x.shape  # T = N + 1 tokens
        # (B, T, 3D) -> 3 x (B, n_heads, T, head_dim)
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Dropout only while training: self.training is set by model.train() / model.eval().
        p = self.dropout if self.training else 0.0

        if return_attention:
            # Explicit version, so we can return the attention weights (for visualisation).
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, T)
            weights = scores.softmax(dim=-1)
            y = F.dropout(weights, p, self.training) @ v  # (B, h, T, head_dim)
        else:
            # Same maths, fused kernel (FlashAttention / memory-efficient on GPU).
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            weights = None

        y = y.transpose(1, 2).reshape(B, T, D)  # concatenate the heads
        return self.out_dropout(self.out_proj(y)), weights


# ---------------------------------------------------------------------------
# 3. Transformer block
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Per-token feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int(config.mlp_ratio * config.d_model)
        self.fc1 = nn.Linear(config.d_model, hidden)
        self.fc2 = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm Transformer block: normalise, transform, add back (residual)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attn(self.ln1(x), return_attention)
        x = x + attended  # tokens exchange information
        x = x + self.mlp(self.ln2(x))  # each token is processed on its own
        return x, weights


# ---------------------------------------------------------------------------
# 4. The full model
# ---------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """Image classifier: patches -> Transformer encoder -> class logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        # A learned token with no image content: through attention it gathers a
        # summary of the whole image, and the head classifies from it.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        # Attention ignores order, so each position (CLS + every patch) gets a learned vector.
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Small random weights (std 0.02) everywhere, a few deliberate exceptions."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Each block adds its output to the residual stream; scale the last layer
        # of each branch down so the stream does not grow with depth (GPT-2 trick).
        for block in self.blocks:
            for layer in (block.attn.out_proj, block.mlp.fc2):
                nn.init.trunc_normal_(layer.weight, std=0.02 / math.sqrt(2 * self.config.n_layers))
        # Zero head: every class starts equally likely, so the first loss is
        # exactly ln(num_classes) (a free sanity check, see tests/test_model.py).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(
        self, images: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """(B, C, H, W) -> normalised tokens (B, N+1, D) and (optionally) each layer's attention."""
        size = self.config.image_size
        if images.shape[-2:] != (size, size):
            raise ValueError(f"expected {size}x{size} images, got {tuple(images.shape[-2:])}")
        x = self.patch_embed(images)  # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # (B, N+1, D)
        x = self.dropout(x)
        attentions = []
        for block in self.blocks:
            x, weights = block(x, return_attention)
            if weights is not None:
                attentions.append(weights)
        return self.norm(x), attentions

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> class logits (B, num_classes)."""
        tokens, _ = self.forward_features(images)
        pooled = tokens[:, 0] if self.config.pool == "cls" else tokens[:, 1:].mean(dim=1)
        return self.head(pooled)

    @torch.no_grad()
    def attention_maps(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Attention weights of every layer, each (B, n_heads, N+1, N+1)."""
        return self.forward_features(images, return_attention=True)[1]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_groups(self, weight_decay: float) -> list[dict]:
        """Two AdamW groups: weight matrices are decayed; biases, norms,
        the [CLS] token and position embeddings are not (decaying them only hurts)."""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and name not in ("cls_token", "pos_embed"):
                decay.append(param)
            else:
                no_decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]


def attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """Which patches did the [CLS] token look at, through all layers? (Abnar & Zuidema, 2020)

    Average the heads, add the identity for the residual connection, renormalise,
    and multiply the layers together. Returns (B, N): one weight per patch,
    summing to 1 (all 0 if [CLS] never looked at a patch). Reshape it to
    (H/P, W/P) to overlay it on the image.
    """
    rollout = None
    for weights in attentions:
        a = weights.mean(dim=1)  # (B, T, T)
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout
    cls_to_patches = rollout[:, 0, 1:]
    # (clamp: if [CLS] only ever attended to itself, every patch weight is 0, not NaN)
    return cls_to_patches / cls_to_patches.sum(dim=-1, keepdim=True).clamp(min=1e-12)
