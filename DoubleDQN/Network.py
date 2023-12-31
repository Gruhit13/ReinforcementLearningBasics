import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple
import numpy as np

class Network(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_shape = self.get_conv_shape(obs_shape).item()

        self.ffn1 = nn.Linear(conv_shape, 512)
        self.ffn2 = nn.Linear(512, n_actions)

    def get_conv_shape(self, inp_shape: Tuple[int, int, int]) -> int:
        # Pytorch model takes input as (C, H, W)
        h, w, c = inp_shape
        inp_shape = (c, h, w)
        rand_input = torch.rand(1, *inp_shape)
        return np.prod(self.conv(rand_input).data.shape)

    def forward(self, state : Tensor) -> Tensor:
        # Forward will be used to learn state action value
        batch_size = state.shape[0]
        state /= 255.
        x = self.conv(state)
        x = x.view(batch_size, -1)
        x = self.ffn2(F.relu(self.ffn1(x)))
        return x