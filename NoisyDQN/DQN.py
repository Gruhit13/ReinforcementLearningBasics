import numpy as np
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Tuple

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if torch.cuda.is_available() else autograd.Variable(*args, **kwargs)

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

        self.noisy1 = NoisyLinear(conv_shape, 512)
        self.noisy2 = NoisyLinear(512, n_actions)

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
        x = self.noisy2(F.relu(self.noisy1(x)))
        return x
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# Based on the paper Noisy Network for Exploration and taking reference from
# the book Deep Reinforcement learning hands on using a NN to mitigate a 
# need for extra parameter for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x