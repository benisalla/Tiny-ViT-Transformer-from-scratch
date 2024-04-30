import torch
from torch.nn import nn


class ViTPosEncoding(nn.Module):
    def __init__(self, n_embd, p_size, im_size):
        super(ViTPosEncoding, self).__init__()

        self.ppe = nn.Parameter(torch.randn((im_size // p_size) **2 + 1, n_embd))

    def forward(self, x):
        x += self.ppe
        return x