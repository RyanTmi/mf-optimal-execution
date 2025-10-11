import argparse, time
from pathlib import Path
from dataclasses import asdict

import torch
import numpy as np
from tqdm import trange

from mf_optimal_execution.models import PriceImpactModel
from mf_optimal_execution.solutions import OptimalExecutionSolutionFBSDE_MFC
from mf_optimal_execution.networks import DirectMFCModel
from mf_optimal_execution.utils import get_device, setup_plt
from mf_optimal_execution.config import get_example_config


# ==============================================================================
# Eval
# ==============================================================================


@torch.no_grad()
def evaluate_model(model: DirectMFCModel, sol_ref: OptimalExecutionSolutionFBSDE_MFC, *, cfg, device) -> tuple[float, float]:
    n, time_step = 1_000, 100
    dt = cfg.maturity / time_step
    x0 = cfg.x0_std * torch.randn((n, 1), device=device) + cfg.x0_mean
    dw = (dt**0.5) * torch.randn((n, time_step, 1), device=device)

    xs, aa = model.build_trajectories(x0, dw)
    t_grid = torch.linspace(0.0, cfg.maturity, time_step + 1, device=device)
    alpha_star = torch.stack([sol_ref.alpha(float(t_grid[i].item()), xs[:, i]) for i in range(time_step + 1)], dim=1)

    err = (alpha_star - aa).pow(2).mean(dim=1)  # per-path MSE over time
    return err.mean().item(), (err.std().item() if err.numel() > 1 else 0.0)


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


def train_mkv(n_epochs: int, lr: float, batch_size: int, time_steps: int, *, outdir: Path, device) -> None:
    setup_plt()
    set_seed(42)

    cfg = get_example_config()
    dyn = PriceImpactModel(cfg.sigma, cfg.gamma, cfg.ca, cfg.cx, cfg.cg)
    sol = OptimalExecutionSolutionFBSDE_MFC(dyn, cfg)

    model = DirectMFCModel(dyn, cfg.maturity).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses, mean_errs, std_errs = [], [], []
    dt = cfg.maturity / time_steps

    for _ in trange(n_epochs, desc="Training Direct MFC"):
        x = cfg.x0_std * torch.randn((batch_size, 1), device=device) + cfg.x0_mean
        dw = (dt**0.5) * torch.randn((batch_size, time_steps, 1), device=device)

        loss = model(x, dw)
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
        outdir / f"direct_mfc_{int(time.time())}.pt",
    )
    print(f"Chekpoint saved at {outdir / f"direct_mfc_{int(time.time())}.pt"}")


def main():
    p = argparse.ArgumentParser(description="Train Direct MFC method.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=10_000)
    p.add_argument("--time-steps", type=int, default=100)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    device = get_device(args.device)
    train_mkv(args.epochs, args.lr, args.batch_size, args.time_steps, outdir=args.outdir, device=device)


if __name__ == "__main__":
    main()
