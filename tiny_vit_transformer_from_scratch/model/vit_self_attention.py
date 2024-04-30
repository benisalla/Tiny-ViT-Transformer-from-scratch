import torch
import torch.nn as nn
import math

class ViTSelfAttention(nn.Module):
    def __init__(self, n_patch, n_embd, h_size, d_rate=0.0, bias=False):
        super(ViTSelfAttention, self).__init__()

        assert n_embd % h_size == 0

        self.n_embd = n_embd
        self.h_size  = h_size
        self.n_head = n_embd // h_size
        self.d_rate = d_rate
        self.n_patch = n_patch

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(d_rate)
        self.resid_dropout = nn.Dropout(d_rate)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(n_patch, n_patch)).view(1, 1, n_patch, n_patch))

    def forward(self, x, training=True):
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
