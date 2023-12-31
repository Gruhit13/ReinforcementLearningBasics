import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class NoisyLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            sigma_init: float = 0.017,
            bias: bool = True
    ) -> None:
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features, ), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    
    def forward(self, input: Tensor) -> Tensor:
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)

class DQN(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_size=64):
		super(DQN, self).__init__()

		self.linear1 = NoisyLinear(input_dim, hidden_size)
		self.linear2 = NoisyLinear(hidden_size, 64)
		self.linear3 = NoisyLinear(64, output_dim)

	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		return self.linear3(x)
		