import argparse, time
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

from mf_optimal_execution.models import PriceImpactModel
from mf_optimal_execution.solutions import OptimalExecutionSolutionFBSDE_MFG
from mf_optimal_execution.networks import FBSDEModel_MFG
from mf_optimal_execution.utils import get_device
from mf_optimal_execution.config import get_example_config


# ==============================================================================
# Eval
# ==============================================================================


@torch.no_grad()
def evaluate_model(
    model: FBSDEModel_MFG,
    sol: OptimalExecutionSolutionFBSDE_MFG,
    *,
    cfg,
    device,
) -> tuple[float, float]:
    n, time_step = 1_000, 100
    dt = cfg.maturity / time_step

    x = cfg.x0_std * torch.randn((n, 1), device=device) + cfg.x0_mean
    dw = (dt**0.5) * torch.randn((n, time_step, 1), device=device)
    xs, ys = model.build_trajectories(x, dw)

    mean_errors = np.zeros(time_step + 1)
    std_errors = np.zeros(time_step + 1)
    for i in range(time_step + 1):
        error = (sol.alpha(i * dt, xs[:, int(i)]) - (-ys[:, int(i)] / cfg.ca)).pow(2)
        mean_errors[i] = error.mean().item()
        std_errors[i] = error.std().item()
    return mean_errors.mean(), std_errors.mean()


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


def train_fbsde(n_epochs: int, lr: float, batch_size: int, time_steps: int, *, outdir: Path, device) -> None:
    set_seed(42)

    cfg = get_example_config()
    dyn = PriceImpactModel(cfg.sigma, cfg.gamma, cfg.ca, cfg.cx, cfg.cg)
    sol = OptimalExecutionSolutionFBSDE_MFG(dyn, cfg)

    model = FBSDEModel_MFG(dyn, cfg.maturity).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    losses, mean_errs, std_errs = [], [], []
    dt = cfg.maturity / time_steps

    for _ in trange(n_epochs, desc="Training FBSDE MFG"):
        x = cfg.x0_std * torch.randn((batch_size, 1), device=device) + cfg.x0_mean
        dw = (dt**0.5) * torch.randn((batch_size, time_steps, 1), device=device)

        y, g = model(x, dw)
        loss = criterion(y, g)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        m, s = evaluate_model(model, sol, cfg=cfg, device=device)
        mean_errs.append(m)
        std_errs.append(s)

    model.cpu()

    losses = torch.tensor(losses)
    mean_errs = torch.tensor(mean_errs)
    std_errs = torch.tensor(std_errs)

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "losses": losses,
            "mean_errors": mean_errs,
            "std_errors": std_errs,
            "config": asdict(cfg),
        },
        outdir / f"fbsde_mfg_{int(time.time())}.pt",
    )
    print(f"Chekpoint saved at {outdir / f"fbsde_mfg_{int(time.time())}.pt"}")


def main():
    p = argparse.ArgumentParser(description="Train FBSDE MFG method.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--time-steps", type=int, default=100)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    device = get_device(args.device)
    train_fbsde(args.epochs, args.lr, args.batch_size, args.time_steps, outdir=args.outdir, device=device)


if __name__ == "__main__":
    main()
