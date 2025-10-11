import argparse, time
from pathlib import Path
from dataclasses import asdict

import torch
import numpy as np
from tqdm import trange

from mf_optimal_execution.models import PriceImpactModel
from mf_optimal_execution.networks import DGMModel
from mf_optimal_execution.utils import get_device
from mf_optimal_execution.config import get_example_config


# ==============================================================================
# Sample
# ==============================================================================


def sample(n_sample: int, s_min: float, s_max: float, device: torch.device) -> torch.Tensor:
    x = (s_max - s_min) * torch.rand((n_sample, 1), device=device) + s_min
    return x


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


def train_dgm(n_epochs: int, lr: float, *, outdir: Path, load_checkpoint: bool, device: torch.device) -> None:
    set_seed(42)

    cfg = get_example_config()
    dyn = PriceImpactModel(cfg.sigma, cfg.gamma, cfg.ca, cfg.cx, cfg.cg)

    model = DGMModel(dyn, cfg.maturity, cfg.x0_mean, cfg.x0_std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    previous_losses = None
    if load_checkpoint:
        candidates = sorted((p for p in outdir.glob("dgm_mfg_*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(candidates) > 0:
            ckpt = torch.load(candidates[0])
            model.load_state_dict(ckpt["model"], strict=True)
            opt.load_state_dict(ckpt["optimizer"])
            previous_losses = ckpt["losses"]
            print(f"Checkpoint '{candidates[0].name}' loaded")

    weights = torch.tensor([2.0, 3.0, 2.0, 1.0], device=device)

    x_min, x_max = -1.0, 1.0

    n_sample_t = 20
    n_sample_x = 500
    n_sample_x_b = 1_000

    train_losses = []
    for epoch in trange(n_epochs, desc="Training DGM MFG"):
        if epoch % 100 == 0:
            t = sample(n_sample_t, 0, cfg.maturity, device)
            x = sample(n_sample_x, x_min, x_max, device)
            x_0 = sample(n_sample_x_b, x_min, x_max, device)
            x_t = sample(n_sample_x_b, x_min, x_max, device)

        losses = model(t, x, x_0, x_t)
        loss = weights.dot(losses)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            train_losses.append(losses.detach().cpu())

    model.cpu()
    train_losses = torch.stack(train_losses)

    if previous_losses is not None:
        train_losses = torch.cat([previous_losses, train_losses], dim=0)

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "losses": train_losses,
            "config": asdict(cfg),
        },
        outdir / f"dgm_mfg_{int(time.time())}.pt",
    )
    print(f"Chekpoint saved at {outdir / f"dgm_mfg_{int(time.time())}.pt"}")


def main():
    p = argparse.ArgumentParser(description="Train DGM MFG method.")
    p.add_argument("--epochs", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--load-checkpoint", type=bool, default=True)
    p.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    device = get_device(args.device)
    train_dgm(args.epochs, args.lr, outdir=args.outdir, load_checkpoint=args.load_checkpoint, device=device)


if __name__ == "__main__":
    main()
