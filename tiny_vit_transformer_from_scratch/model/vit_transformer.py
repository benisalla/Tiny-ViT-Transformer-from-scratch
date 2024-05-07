from tiny_vit_transformer_from_scratch.model.vit_block import ViTBlock
from tiny_vit_transformer_from_scratch.model.vit_embedding import ViTConv2dEmbedding, ViTLNEmbedding
from tiny_vit_transformer_from_scratch.model.vit_layernorm import ViTLayerNorm
from tiny_vit_transformer_from_scratch.model.vit_pos_encoding import ViTPosEncoding
from tiny_vit_transformer_from_scratch.core.config import VitConfig
import torch
import torch.nn as nn
import math


class VisionTransformer(nn.Module):
    
    """
    A Vision Transformer (ViT) model implementation utilizing PyTorch. This class encapsulates
    the entire architecture of a ViT, including embeddings, positional encodings, transformer
    blocks, and layer normalization.

    Attributes:
        n_block (int): Number of transformer blocks.
        n_embd (int): Dimensionality of the embeddings.
        h_size (int): Dimensionality of the multi-head attention heads.
        p_size (int): Size of each image patch.
        im_size (int): Size of the input images.
        c_dim (int): Channel dimension of the input.
        n_class (int): Number of output classes.
        d_rate (float): Dropout rate.
        bias (bool): Whether to use bias in the linear layers.
        h_dim (int): Hidden dimensionality, dependent on c_dim.
        encoder (nn.ModuleDict): A module dict containing all the transformer components.
        lm_head (nn.Linear): Output linear layer that maps transformer output to class logits.

    Methods:
        get_num_params (bool): Calculates the total number of trainable parameters in the model.
        _init_weights (module): Applies a specific initialization to the weights of various modules
                                in the transformer based on the module type.
        forward (x): Defines the forward pass of the Vision Transformer.
    """

    def __init__(self, config: VitConfig):
        super(VisionTransformer, self).__init__()
        """
        Initializes the Vision Transformer model with the specified configuration.

        Args:
            config (VitConfig): Configuration object containing attributes like number of blocks,
                                dimensions of embeddings, etc.
        """
        super(VisionTransformer, self).__init__()
        assert config.n_block is not None and config.p_size is not None and \
                config.embd_dim is not None and config.head_dim is not None, \
                "Configuration must include n_block, p_size, embd_dim, and head_dim."

        # Model parameters and architecture configuration
        self.n_block = config.n_block
        self.n_embd = config.n_embd
        self.h_size = config.h_size
        self.p_size = config.p_size
        self.im_size = config.im_size
        self.c_dim = config.c_dim
        self.n_class = config.n_class
        self.d_rate = config.d_rate
        self.bias = config.bias
        self.n_patch = config.im_size**2 / config.p_size**2
        self.h_dim = config.h_dim if config.c_dim else 4 * config.n_embd

        self.encoder = nn.ModuleDict(dict(
            # pte = ViTConv2dEmbedding(n_embd=self.n_embd, p_size=self.p_size, c_dim=self.c_dim),
            pte = ViTLNEmbedding(n_embd=self.n_embd, p_size=self.p_size, c_dim=self.c_dim),
            ppe = ViTPosEncoding(n_embd=self.n_embd, p_size=self.p_size, im_size=self.im_size),
            dropout = nn.Dropout(self.d_rate),
            blocks = nn.ModuleList([
                ViTBlock(
                    n_patch=self.n_patch,
                    n_embd=self.n_embd,
                    h_size=self.h_size,
                    h_dim=self.h_dim,
                    d_rate=self.d_rate,
                    bias=self.bias)
                for _ in range(self.n_block)]),
            ln = ViTLayerNorm(self.n_embd, bias=self.bias),
        ))

        self.lm_head = nn.Linear(self.n_embd, self.n_class, bias=self.bias)
        # self.encoder.pte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        for par_name, par in self.named_parameters():
            if par_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(par, mean=0.0, std=0.02/math.sqrt(2 * self.n_block))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        
    def get_num_params(self):
        """
        Returns the total number of trainable parameters in the model.

        Args:
            non_embedding (bool): If True, excludes embedding parameters from the count.

        Returns:
            int: Total number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    
    def _init_weights(self, module):
        """
        Applies initial weights to certain types of layers within the model.

        Args:
            module (nn.Module): The module to potentially initialize.
        """
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
            
    def forward(self, x):
        """
        Defines the forward pass of the Vision Transformer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: The output logits of the Vision Transformer.
        """
        
        x = self.encoder.pte(x)
        x = self.encoder.ppe(x)

        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.ln(x)

        cls = x[:, 0, :]
        logits = self.lm_head(cls)

        return logits