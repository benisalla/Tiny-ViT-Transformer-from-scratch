from tiny_vit_transformer_from_scratch.model.vit_feedforward import ViTFeedForwardMLP
from tiny_vit_transformer_from_scratch.model.vit_layernorm import ViTLayerNorm
from tiny_vit_transformer_from_scratch.model.vit_self_attention import ViTSelfAttention


class ViTBlock(nn.Module):
    def __init__(self, n_patch, n_embd, h_size, h_dim=None, d_rate=0.0, bias=False):
        super(ViTBlock, self).__init__()
        self.attn_ln = ViTLayerNorm(n_dim=n_embd, bias=bias)
        self.attn = ViTSelfAttention(n_patch=n_patch, n_embd=n_embd, h_size=h_size, d_rate=d_rate, bias=bias)
        self.ff_ln = ViTLayerNorm(n_dim=n_embd, bias=bias)
        self.ff_mlp = ViTFeedForwardMLP(n_embd=n_embd, h_dim=h_dim, d_rate=d_rate, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.attn_ln(x))
        x = x + self.ff_mlp(self.ff_ln(x))
        return x