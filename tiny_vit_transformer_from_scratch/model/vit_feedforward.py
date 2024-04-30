from torch.nn import nn
from tiny_vit_transformer_from_scratch.model.vit_glu_act_fun import ViTGELUActFun


class ViTFeedForwardMLP(nn.Module):
    def __init__(self, n_embd, h_dim=None, d_rate=0.0, bias=False):
        super(ViTFeedForwardMLP, self).__init__()

        if not h_dim:
            h_dim = 4 * n_embd

        self.vit_mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(), # ViTGELUActFun()
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(d_rate),
            )

    def forward(self, x):
        x = self.vit_mlp(x)
        return x