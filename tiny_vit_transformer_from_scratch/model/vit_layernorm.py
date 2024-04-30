import torch
from torch.nn import nn
import torch.nn.functional as F


class ViTLayerNorm(nn.Module):
    def __init__(self, n_dim, bias=False):
        super(ViTLayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(n_dim))
        self.b = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.w.shape, self.w, self.b, 1e-5)