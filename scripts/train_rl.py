import argparse, time
from pathlib import Path
from dataclasses import asdict

import numpy as np
from tqdm import trange

from mf_optimal_execution.models import PriceImpactModel
from mf_optimal_execution.config import Config, get_example_config


# ==============================================================================
# Sample
# ==============================================================================


def generate_datas(n_particules: int, time_steps: int, dt: float, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    x = cfg.x0_mean + cfg.x0_std * np.random.randn(n_particules)
    dw = (dt**0.5) * np.random.randn(n_particules, time_steps)
    return x, dw


def index_of(space: np.ndarray, x: np.ndarray) -> int:
    x_idx = np.argmin(np.abs(space - x)).item()
    return int(x_idx)


def clamp_and_index(space: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int]:
    x_clamped = np.clip(x, space[0], space[-1])
    x_idx = index_of(space, x_clamped)
    return x_clamped, x_idx


def choose_action(action_space, action_size, q, n, x_idx, eps) -> np.ndarray:
    if np.random.rand() > eps:
        return action_space[np.random.randint(action_size, size=1)]
    else:
        return action_space[np.argmin(q[n, x_idx])]


# ==============================================================================
# Training loop
# ==============================================================================


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def train_rl(n_epochs: int, *, outdir: Path) -> None:
    set_seed(42)

    cfg = get_example_config()
    dyn = PriceImpactModel(cfg.sigma, cfg.gamma, cfg.ca, cfg.cx, cfg.cg)

    time_steps = 16
    dt = cfg.maturity / time_steps

    step = dt**0.5
    action_space = np.arange(-2.5, 1 + step, step)
    state_space = np.arange(-1.5, 1.75 + step, step)

    action_size = action_space.shape[0]
    state_size = state_space.shape[0]

    w_q, w_t = 0.55, 0.85

    shape = (time_steps + 1, state_size, action_size)

    q = np.zeros(shape)
    v = np.full(shape, fill_value=1 / (state_size * action_size))
    h = np.zeros(shape, dtype=np.int32)
    d = np.zeros_like(v[0])

    eps = 0.1
    for k in trange(n_epochs, desc="Train MF Q-Learning"):
        if k % 20000 == 0:
            xs, dws = generate_datas(20000, time_steps, dt, cfg)

        v_old = v.copy()
        q_old = q.copy()

        x, dw = xs[k % 20000], dws[k % 20000]
        x, x_idx = clamp_and_index(state_space, x)

        rho_t = pow(2 + k, -w_t)

        for n in range(time_steps + 1):
            # Choose action using epsilon-greedy policy
            a = choose_action(action_space, action_size, q_old, n, x_idx, eps)
            a_idx = index_of(action_space, a)

            h[n, x_idx, a_idx] += 1
            rho_q = pow(1 + time_steps * h[n, x_idx, a_idx], -w_q)

            # Update population/action distribution
            d[:] = 0
            d[x_idx, a_idx] = 1.0
            v[n] = v_old[n] + rho_t * (d - v_old[n])

            # Observe cost and state
            if n < time_steps:
                x_next = x + dyn.drift(a) * dt + dyn.diffusion() * dw[n]
                x_next, x_next_idx = clamp_and_index(state_space, x_next)

                cost = dyn.running_cost(x, a, (v[n] * action_space).sum()) * dt
                cost = cost + np.min(q_old[n + 1, x_next_idx])
            else:
                cost = dyn.terminal_cost(x)
            q_val = q_old[n, x_idx, a_idx]
            q[n, x_idx, a_idx] = q_val + rho_q * (cost.item() - q_val)

            x, x_idx = x_next, x_next_idx

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(
        outdir / f"rl_mfg_{int(time.time())}.npy",
        {
            "q": q,
            "v": v,
            "state_space": state_space,
            "action_space": action_space,
            "time_steps": time_steps,
            "config": asdict(cfg),
        },
        allow_pickle=True,
    )
    print(f"Chekpoint saved at {outdir / f"rl_mfg_{int(time.time())}.npy"}")


def main():
    p = argparse.ArgumentParser(description="Train MF Q-Learning method.")
    p.add_argument("--epochs", type=int, default=5_000_000)
    p.add_argument("--outdir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    train_rl(args.epochs, outdir=args.outdir)


if __name__ == "__main__":
    main()
