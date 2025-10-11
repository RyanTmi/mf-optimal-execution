import argparse, time
from pathlib import Path
from dataclasses import asdict

import torch
import numpy as np
from tqdm import trange

from mf_optimal_execution.models import PriceImpactModel
from mf_optimal_execution.solutions import OptimalExecutionSolutionPDE_MFG
from mf_optimal_execution.networks import Actor, Critic, Score
from mf_optimal_execution.utils import get_device
from mf_optimal_execution.config import Config, get_example_config


# ==============================================================================
# Environment
# ==============================================================================


class Environment:
    def __init__(self, batch_size: int, n_steps: int, dyn: PriceImpactModel, cfg: Config, device: torch.device):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.dyn = dyn
        self.maturity = cfg.maturity
        self.x0_mean = cfg.x0_mean
        self.x0_std = cfg.x0_std
        self.dt = cfg.maturity / n_steps
        self.sqrt_dt = np.sqrt(self.dt)
        self.device = device

    def reset(self) -> torch.Tensor:
        self.step_coutner = 0
        self.dw = self.sqrt_dt * torch.randn((self.batch_size, self.n_steps, 1), device=self.device)
        self.x = self.x0_mean + torch.randn((self.batch_size, 1), device=self.device) * self.x0_std
        self.t = torch.zeros_like(self.x)
        return self.t, self.x

    def step(self, a: torch.Tensor, nu: torch.Tensor):
        cost = self.dyn.running_cost(self.x, a, nu.mean()) * self.dt
        dx = self.dyn.drift(a) * self.dt + self.dyn.diffusion() * self.dw[:, self.step_coutner]

        self.x = self.x + dx
        self.t = self.t + self.dt
        self.step_coutner += 1

        done = self.step_coutner >= self.n_steps
        return self.t, self.x, cost, done


# ==============================================================================
# Evaluate
# ==============================================================================


@torch.no_grad()
def evaluate_model(
    actor: Actor,
    critic: Critic,
    score: Score,
    cfg: Config,
    sol: OptimalExecutionSolutionPDE_MFG,
    device: torch.device,
):
    n_sample, time_step = 500, 50
    dt = cfg.maturity / time_step

    mean_errors = torch.zeros((time_step + 1, 3))
    std_errors = torch.zeros((time_step + 1, 3))
    for i in range(time_step + 1):
        mu_x, var_x = sol.x_dist(i * dt)
        mu_a, var_a = sol.a_dist(i * dt)
        xx = mu_x + np.sqrt(var_x) * torch.randn((n_sample, 1), device=device)
        aa = mu_a + np.sqrt(var_a) * torch.randn((n_sample, 1), device=device)
        tt = torch.full_like(xx, fill_value=i * dt)

        with torch.enable_grad():
            aa.requires_grad_(True)
            log_p = torch.log(sol.a_pdf(i * dt, aa))
            grad_log_p = torch.autograd.grad(log_p, aa, grad_outputs=torch.ones_like(aa))[0]

        actor_error = (sol.alpha(i * dt, xx) - actor(tt, xx)[0]).pow(2)
        critic_error = (sol.v(i * dt, xx) - critic(tt, xx)).pow(2)
        score_error = (grad_log_p - score(tt, aa)).pow(2)

        mean_errors[i] = torch.tensor([actor_error.mean(), critic_error.mean(), score_error.mean()])
        std_errors[i] = torch.tensor([actor_error.std(), critic_error.std(), score_error.std()])
    return mean_errors.mean(dim=0), std_errors.mean(dim=0)


# ==============================================================================
# Training loop
# ==============================================================================


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)


def langevin_sample_action(
    score: Score,
    n_mf_sample: int,
    steps: int,
    t: float,
    eps: float,
    device: torch.device,
) -> torch.Tensor:
    sqrt_eps = np.sqrt(eps)
    z = torch.randn((steps, n_mf_sample, 1), device=device)
    a = torch.randn((n_mf_sample, 1), device=device)
    t = torch.full_like(a, fill_value=t)
    for i in range(steps):
        a = a + 0.5 * eps * score(t, a) + sqrt_eps * z[i]
    return a


def sample_action(actor: Actor, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, std = actor(t, x)
    a = std * torch.randn_like(x) + mu
    logp = -0.5 * (torch.log(2 * np.pi * std**2) + ((a - mu) / std) ** 2)
    return a, logp


def train_drl(n_episodes: int, *, outdir: Path, load_checkpoint: bool, device: torch.device) -> None:
    set_seed(42)

    cfg = get_example_config()
    dyn = PriceImpactModel(cfg.sigma, cfg.gamma, cfg.ca, cfg.cx, cfg.cg)
    sol = OptimalExecutionSolutionPDE_MFG(dyn, cfg)
    env = Environment(batch_size=32, n_steps=100, dyn=dyn, cfg=cfg, device=device)

    actor = Actor().to(device)
    critic = Critic().to(device)
    score = Score().to(device)

    actor_lr = 5e-6
    critic_lr = 1e-5
    score_lr = 1e-6

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=critic_lr)
    score_opt = torch.optim.AdamW(score.parameters(), lr=score_lr)

    langevin_steps = 200
    n_mf_sample = 1_000
    eps = 0.05

    train_losses, mean_errors, std_errors = [], [], []
    for episode in trange(n_episodes, desc="Train MF Actor-Critc"):
        t, x = env.reset()
        for n in range(env.n_steps):
            a, logp = sample_action(actor, t, x)

            a_score = a.detach()
            a_score.requires_grad_(True)
            s = score(t, a_score)
            grad_a_score = torch.autograd.grad(s, a_score, grad_outputs=torch.ones_like(a_score), create_graph=True)[0]

            score_loss = (grad_a_score + 0.5 * s**2).mean()
            score_opt.zero_grad(set_to_none=True)
            score_loss.backward()
            score_opt.step()

            with torch.no_grad():
                nu = langevin_sample_action(score, n_mf_sample, langevin_steps, n * env.dt, eps, device)
                t_next, x_next, cost, done = env.step(a, nu)
                v_next = dyn.terminal_cost(x_next) if done else critic(t_next, x_next)

            v = critic(t, x)
            td_target = (cost + v_next).detach()
            td_error = td_target - v

            critic_loss = td_error.pow(2).mean()
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_opt.step()

            adv = td_error.detach()
            actor_loss = (logp * adv).mean()
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()

            t, x = t_next, x_next

        if (episode + 1) % 10 == 0:
            losses = torch.stack([score_loss, critic_loss, actor_loss])
            train_losses.append(losses.detach().cpu())
            m, s = evaluate_model(actor, critic, score, cfg, sol, device)
            mean_errors.append(m)
            std_errors.append(s)

    actor.cpu()
    critic.cpu()
    score.cpu()
    train_losses = torch.stack(train_losses)
    mean_errors = torch.stack(mean_errors)
    std_errors = torch.stack(std_errors)

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "score": score.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "score_opt": score_opt.state_dict(),
            "losses": train_losses,
            "mean_errors": mean_errors,
            "std_errors": std_errors,
            "config": asdict(cfg),
        },
        outdir / f"drl_mfg_{int(time.time())}.pt",
    )
    print(f"Chekpoint saved at {outdir / f"drl_mfg_{int(time.time())}.pt"}")


def main():
    p = argparse.ArgumentParser(description="Train MF Actor-Critic MFG method.")
    p.add_argument("--episodes", type=int, default=1_000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--load-checkpoint", type=bool, default=True)
    p.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    device = get_device(args.device)
    train_drl(args.episodes, outdir=args.outdir, load_checkpoint=args.load_checkpoint, device=device)


if __name__ == "__main__":
    main()
