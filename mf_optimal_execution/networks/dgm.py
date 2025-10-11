import torch
import torch.nn as nn

from ..models import PriceImpactModel


class DGMLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation=torch.tanh,
    ):
        super().__init__()
        self.__act = activation

        self.__u_z = nn.Linear(input_dim, hidden_dim, bias=True)
        self.__w_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.__u_g = nn.Linear(input_dim, hidden_dim, bias=True)
        self.__w_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.__u_r = nn.Linear(input_dim, hidden_dim, bias=True)
        self.__w_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.__u_h = nn.Linear(input_dim, hidden_dim, bias=True)
        self.__w_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.__reset_parameters()

    def forward(self, x: torch.Tensor, sl: torch.Tensor) -> torch.Tensor:
        z = self.__act(self.__u_z(x) + self.__w_z(sl))
        g = self.__act(self.__u_g(x) + self.__w_g(sl))
        r = self.__act(self.__u_r(x) + self.__w_r(sl))
        h = self.__act(self.__u_h(x) + self.__w_h(sl * r))
        return (1.0 - g) * h + z * sl

    def __reset_parameters(self) -> None:
        for m in [self.__u_z, self.__w_z, self.__u_g, self.__w_g, self.__u_r, self.__w_r, self.__u_h, self.__w_h]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class DGMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        activation=torch.tanh,
    ):
        super().__init__()
        self.__act = activation

        self.__w_1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.__blocks = nn.ModuleList([DGMLayer(input_dim, hidden_dim, activation=activation) for _ in range(n_layers)])
        self.__out = nn.Linear(hidden_dim, output_dim, bias=True)

        self.__reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.__act(self.__w_1(x))
        for block in self.__blocks:
            s = block(x, s)
        return self.__out(s)

    def __reset_parameters(self) -> None:
        for m in [self.__w_1, self.__out]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class VNetwork(nn.Module):
    def __init__(self, model: PriceImpactModel):
        super().__init__()
        self.model = model
        self.net = DGMNet(input_dim=2, output_dim=1, hidden_dim=50, n_layers=3, activation=torch.tanh)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=1))

    def terminal_value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.model.cg * x**2


class UNetwork(nn.Module):
    def __init__(self, x0_mean: float, x0_std: float):
        super().__init__()
        self.x0_mean = x0_mean
        self.x0_std = x0_std
        self.net = DGMNet(input_dim=2, output_dim=1, hidden_dim=50, n_layers=3, activation=torch.tanh)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=1))

    def initial_value(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x0_mean) ** 2 / (2 * self.x0_std**2)


class DGMModel(nn.Module):
    def __init__(self, model: PriceImpactModel, maturity: float, x0_mean: float, x0_std: float):
        super().__init__()
        self.model = model
        self.maturity = maturity
        self.v_net = VNetwork(model)
        self.u_net = UNetwork(x0_mean, x0_std)

    def forward(self, t: torch.Tensor, x: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        m = self.model

        n_t, n_x = t.size(0), x.size(0)
        t_grid = t.repeat_interleave(n_x, dim=0).requires_grad_(True)
        x_grid = x.repeat(n_t, 1).requires_grad_(True)

        u = self.u_net(t_grid, x_grid)
        u_t = torch.autograd.grad(u, t_grid, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_grid, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_grid, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        v = self.v_net(t_grid, x_grid)
        v_t = torch.autograd.grad(v, t_grid, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x_grid, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x_grid, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

        u_b, ut_b, vx_b = u.view(n_t, n_x, 1), u_t.view(n_t, n_x, 1), v_x.view(n_t, n_x, 1)

        w = torch.softmax(-u_b.squeeze(2), dim=1).unsqueeze(2)
        expand = lambda z: z.repeat_interleave(n_x, dim=0)
        cc = expand(torch.sum(-ut_b * w, dim=1))
        mu = expand(torch.sum(-vx_b * w, dim=1)) / m.ca

        kfp_residual = u_t - (0.5 * m.sigma**2) * (u_xx - u_x**2) + (1 / m.ca) * (v_xx - u_x * v_x) + cc
        hjb_residual = v_t - (m.gamma * x_grid * mu) + (0.5 * m.cx * x_grid**2) + (0.5 * m.sigma**2 * v_xx) - (0.5 * v_x**2 / m.ca)
        kfp_initial = self.u_net(torch.zeros_like(x_0), x_0) - self.u_net.initial_value(x_0)
        hjb_terminal = self.v_net(torch.full_like(x_t, self.maturity), x_t) - self.v_net.terminal_value(x_t)

        hjb_residual_loss = hjb_residual.pow(2).mean().sqrt()
        hjb_boundary_loss = hjb_terminal.pow(2).mean().sqrt()
        kfp_residual_loss = kfp_residual.pow(2).mean().sqrt()
        kfp_boundary_loss = kfp_initial.pow(2).mean().sqrt()
        return torch.stack([hjb_residual_loss, hjb_boundary_loss, kfp_residual_loss, kfp_boundary_loss])
