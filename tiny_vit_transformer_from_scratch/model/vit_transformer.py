from tiny_vit_transformer_from_scratch.model.vit_block import ViTBlock
from tiny_vit_transformer_from_scratch.model.vit_embedding import ViTConv2dEmbedding, ViTLNEmbedding
from tiny_vit_transformer_from_scratch.model.vit_layernorm import ViTLayerNorm
from tiny_vit_transformer_from_scratch.model.vit_pos_encoding import ViTPosEncoding
import torch
from torch.nn import nn
import math


class VisionTransformer(nn.Module):

    def __init__(self, n_block, n_embd, h_size, p_size, im_size, c_dim, n_class, h_dim=None, d_rate=0.0, bias=False):
        super(VisionTransformer, self).__init__()
        assert n_block is not None and p_size is not None and n_embd is not None and h_size is not None
        self.n_block = n_block
        self.n_patch = im_size**2 / p_size**2
        self.n_embd = n_embd
        self.h_size = h_size
        self.p_size = p_size
        self.im_size = im_size
        self.c_dim = c_dim
        self.n_class = n_class
        self.h_dim = h_dim if c_dim else 4 * n_embd
        self.d_rate = d_rate
        self.bias = bias

        self.encoder = nn.ModuleDict(dict(
            # pte = ViTConv2dEmbedding(n_embd=n_embd, p_size=p_size, c_dim=c_dim),
            pte = ViTLNEmbedding(n_embd=n_embd, p_size=p_size, c_dim=c_dim),
            ppe = ViTPosEncoding(n_embd=n_embd, p_size=p_size, im_size=im_size),
            dropout = nn.Dropout(d_rate),
            blocks = nn.ModuleList([
                ViTBlock(
                    n_patch=self.n_patch,
                    n_embd=self.n_embd,
                    h_size=self.h_size,
                    h_dim=self.h_dim,
                    d_rate=self.d_rate,
                    bias=self.bias)
                for _ in range(n_block)]),
            ln = ViTLayerNorm(n_embd, bias=bias),
        ))

        self.lm_head = nn.Linear(n_embd, n_class, bias=bias)
        # self.encoder.pte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        for par_name, par in self.named_parameters():
            if par_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(par, mean=0.0, std=0.02/math.sqrt(2 * n_block))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    
    def _init_weights(self, module):
        std = 0.02  
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, ViTConv2dEmbedding):
            torch.nn.init.normal_(module.ite[0].weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.ite[0].bias)
            torch.nn.init.normal_(module.cls_token, mean=0.0, std=std)
        elif isinstance(module, ViTPosEncoding):
            torch.nn.init.normal_(module.ppe, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)    
            
            
    def get_init_args(self):
        return {
            'n_block': self.n_block,
            'n_embd': self.n_embd,
            'h_size': self.h_size,
            'p_size': self.p_size,
            'im_size': self.im_size,
            'c_dim': self.c_dim,
            'n_class': self.n_class,
            'h_dim': self.h_dim,
            'd_rate': self.d_rate,
            'bias': self.bias
        }
            
    def forward(self, x, y=None):
        device = x.device
        B, C, H, W = x.shape

        x = self.encoder.pte(x)
        x = self.encoder.ppe(x)

        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.ln(x)

        cls = x[:, 0, :]
        logits = self.lm_head(cls)

        return logits