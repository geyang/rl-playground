import numpy as np
import math
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus

def exponential_decaying_epsilon(epsilon_start, epsilon_end, epsilon_decay, t, min_replay_history):
    if t < min_replay_history:
        return epsilon_start
    t = t - min_replay_history
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * t / epsilon_decay)


class MLP(nn.Module):
    def __init__(
            self,
            layers,
            activation=torch.tanh,
            output_activation=None,
            output_scale=1,
            output_squeeze=False,
            fourier_features=False,
            fourier_size=-1,
            fourier_sigma=-1,
            device='cpu',
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze
        self.fourier_features = fourier_features

        if self.fourier_features:
            device = torch.device(device)
            self.fourier_matrix = torch.normal(mean=0.0, std=fourier_sigma, size=(layers[0], fourier_size)).to(device)
            layers[0] = 2*fourier_size
        else:
            self.fourier_matrix = None

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs

        if self.fourier_features:
            x = torch.cat([torch.cos(2 * np.pi * torch.matmul(x, self.fourier_matrix)),
                           torch.sin(2 * np.pi * torch.matmul(x, self.fourier_matrix))],
                           dim=1)

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale

        return x.squeeze() if self.output_squeeze else x


class QMlp(nn.Module):
    """
    in_features should be the sum of all inputs feature dimensions
    to the forward pass.

    """

    def __init__(
            self,
            in_features,
            action_space,
            hidden_sizes=(400, 300),
            activation=torch.relu,
            output_activation=None,
            fourier_features=False,
            fourier_size=-1,
            fourier_sigma=-1,
            device='cpu',
    ):
        super(QMlp, self).__init__()

        assert isinstance(action_space, Discrete), "only supports Discrete action space"
        action_dim = action_space.n

        self.q = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
            output_activation=output_activation,
            fourier_features=fourier_features,
            fourier_size=fourier_size,
            fourier_sigma=fourier_sigma,
            device=device,
        )

    def forward(self, *inputs):
        regular = torch.broadcast_tensors(*inputs)
        block = torch.cat(regular, axis=-1)
        return self.q(block)

    def policy(self, *inputs):
        return torch.argmax(self.q(*inputs), dim=1, keepdim=True)
