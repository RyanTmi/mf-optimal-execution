import torch
import torch.nn as nn

from . import FeedForward


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FeedForward(input_dim=2, output_dim=1, hidden_dim=128, n_hidden=3)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=1))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(2, 128), nn.Tanh())
        self.mu_head = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.std_head = nn.Sequential(nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softplus())

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([t, x], dim=1))
        mu = self.mu_head(h)
        std = self.std_head(h) + 5e-2  # add exploration
        return mu, std


class Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FeedForward(input_dim=2, output_dim=1, hidden_dim=128, n_hidden=3)

    def forward(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, a], dim=1))
