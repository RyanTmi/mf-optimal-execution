import torch
import torch.nn as nn

from . import FeedForward
from ..models import PriceImpactModel


class DirectMFCModel(nn.Module):
    def __init__(self, model: PriceImpactModel, maturity: float) -> None:
        super().__init__()
        self.model = model
        self.maturity = maturity
        self.alpha = FeedForward(
            input_dim=2,
            output_dim=1,
            hidden_dim=128,
            n_hidden=3,
            activation=torch.relu,
        )

    def forward(self, x: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = dw.shape
        dt = self.maturity / time_steps

        t = torch.zeros_like(x)
        cost = torch.zeros_like(x)
        for i in range(time_steps):
            a = self.alpha(torch.cat([t, x], dim=1))

            x_next = x + self.model.drift(a) * dt + self.model.diffusion() * dw[:, i]
            cost = cost + self.model.running_cost(x, a, a.mean()) * dt

            t, x = t + dt, x_next

        cost = cost + self.model.terminal_cost(x)
        return cost.mean()

    @torch.no_grad()
    def build_trajectories(self, x: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        n_sample, time_steps, _ = dw.shape
        dt = self.maturity / time_steps

        xs = torch.empty((n_sample, time_steps + 1))
        aa = torch.empty((n_sample, time_steps + 1))
        t = torch.zeros_like(x)
        xs[:, 0] = x.squeeze()
        for i in range(time_steps):
            a = self.alpha(torch.cat([t, x], dim=1))

            x_next = x + self.model.drift(a) * dt + self.model.diffusion() * dw[:, i]
            t, x = t + dt, x_next
            xs[:, i + 1], aa[:, i] = x.squeeze(), a.squeeze()

        aa[:, -1] = self.alpha(torch.cat([t, x], dim=1)).squeeze()
        return xs, aa
