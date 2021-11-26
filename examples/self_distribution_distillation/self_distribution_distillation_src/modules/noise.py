import torch
import torch.nn as nn
import numpy as np

from self_distribution_distillation_src.utils.device import (
    check_device
)


class MultiplicativeGaussianLayer(nn.Module):
    def __init__(self, a: float = 0.0, b: float = 0.0, use_gpu: bool = True):
        super(MultiplicativeGaussianLayer, self).__init__()
        self.device = check_device(use_gpu)
        self.a, self.b = a, b

        # Fixed standard deviation
        self.use_uniform: bool = np.abs(b - a) > 1e-3

    def get_uniform(self, x: torch.Tensor, *args, **kwargs):
        """
        Sample uniform standard deviation:
            x: (batch, len, vocab)
            u: (batch, len, 1)
        """

        if not self.use_uniform: return self.a

        # Sample noise of data type:
        dtype = x.dtype

        # Get size of noise
        noise_size = x.size()[:-1]

        # Get uniform noise
        noise = torch.rand(*noise_size, dtype = dtype, device = self.device).unsqueeze(-1)

        # Scale the return
        return self.a +  (self.b - self.a) * noise

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        # Sample noise of data type:
        dtype = x.dtype

        # Get noise of chosen data type
        noise = torch.randn(*x.size(), dtype = dtype, device = self.device)

        # Scale noise by a uniform random variable
        noise = noise * self.get_uniform(x)

        # One mean gaussian noise with random std deviation
        return x * (noise + 1.0)
