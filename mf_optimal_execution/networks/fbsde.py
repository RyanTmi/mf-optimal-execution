import torch
import torch.nn as nn

from . import FeedForward
from ..models import PriceImpactModel


class FBSDEModel_MFG(nn.Module):
    def __init__(self, model: PriceImpactModel, maturity: float):
        super().__init__()
        self.model = model
        self.maturity = maturity
        self.y = FeedForward(
            input_dim=1,
            output_dim=1,
            hidden_dim=64,
            n_hidden=2,
            activation=torch.relu,
        )
        self.z = FeedForward(
            input_dim=2,
            output_dim=1,
            hidden_dim=128,
            n_hidden=2,
            activation=torch.relu,
        )

    def forward(self, x: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = dw.shape
        dt = self.maturity / time_steps

        y = self.y(x)
        t = torch.zeros_like(x)
        for i in range(time_steps):
            a = -y / self.model.ca

            x_drift = self.model.drift(a)
            x_diffusion = self.model.diffusion()
            y_drift = -self.model.running_cost_dx(x, a.mean())
            y_diffusion = self.z(torch.cat([t, x], dim=1))

            x = x + x_drift * dt + x_diffusion * dw[:, i]
            y = y + y_drift * dt + y_diffusion * dw[:, i]
            t = t + dt

        return y, self.model.terminal_cost_dx(x)

    @torch.no_grad()
    def build_trajectories(self, x: torch.Tensor, dw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_sample, time_steps, _ = dw.shape
        dt = self.maturity / time_steps

        xs = torch.empty((n_sample, time_steps + 1))
        ys = torch.empty((n_sample, time_steps + 1))
        t = torch.zeros_like(x)
        y = self.y(x)
        xs[:, 0] = x.squeeze()
        ys[:, 0] = y.squeeze()
        for i in range(time_steps):
            a = -y / self.model.ca
            x_drift, x_diffusion = self.model.drift(a), self.model.diffusion()
            y_drift, y_diffusion = -self.model.running_cost_dx(x, a.mean()), self.z(torch.cat([t, x], dim=1))
            x_next = x + x_drift * dt + x_diffusion * dw[:, i]
            y_next = y + y_drift * dt + y_diffusion * dw[:, i]

            x, y, t = x_next, y_next, t + dt
            xs[:, i + 1], ys[:, i + 1] = x.squeeze(), y.squeeze()

        return xs, ys


class FBSDEModel_MFC(nn.Module):
    def __init__(self, model: PriceImpactModel, maturity: float) -> None:
        super().__init__()
        self.model = model
        self.maturity = maturity
        self.y = FeedForward(
            input_dim=1,
            output_dim=1,
            hidden_dim=64,
            n_hidden=2,
            activation=torch.relu,
        )
        self.z = FeedForward(
            input_dim=2,
            output_dim=1,
            hidden_dim=128,
            n_hidden=2,
            activation=torch.relu,
        )

    def forward(self, x: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = dw.shape
        dt = self.maturity / time_steps

        y = self.y(x)
        t = torch.zeros_like(x)
        for i in range(time_steps):
            a = -(y - self.model.gamma * x.mean()) / self.model.ca

            x_drift = self.model.drift(a)
            x_diffusion = self.model.diffusion()
            y_drift = -self.model.running_cost_dx(x, a.mean())
            y_diffusion = self.z(torch.cat([t, x], dim=1))

            x = x + x_drift * dt + x_diffusion * dw[:, i]
            y = y + y_drift * dt + y_diffusion * dw[:, i]
            t = t + dt

        return y, self.model.terminal_cost_dx(x)

    @torch.no_grad()
    def build_trajectories(self, x: torch.Tensor, dw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_sample, time_steps, _ = dw.shape
        dt = self.maturity / time_steps

        xs = torch.empty((n_sample, time_steps + 1))
        ys = torch.empty((n_sample, time_steps + 1))
        t = torch.zeros_like(x)
        y = self.y(x)
        xs[:, 0] = x.squeeze()
        ys[:, 0] = y.squeeze()
        for i in range(time_steps):
            a = -(y - self.model.gamma * x.mean()) / self.model.ca
            x_drift, x_diffusion = self.model.drift(a), self.model.diffusion()
            y_drift, y_diffusion = -self.model.running_cost_dx(x, a.mean()), self.z(torch.cat([t, x], dim=1))
            x_next = x + x_drift * dt + x_diffusion * dw[:, i]
            y_next = y + y_drift * dt + y_diffusion * dw[:, i]

            x, y, t = x_next, y_next, t + dt
            xs[:, i + 1], ys[:, i + 1] = x.squeeze(), y.squeeze()

        return xs, ys
