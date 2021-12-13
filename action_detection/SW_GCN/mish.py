import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This implementation is based on: https://github.com/DelamareMicka/SW-GCN
"""


class Mish(nn.Module):  # basically a softmax layer
    def init(self):
        super().init()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))
