import torch
import matplotlib.pyplot as plt

from typing import Literal


def get_device(prefer: Literal["auto", "cpu", "cuda", "mps"] = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "mps":
        return torch.device("mps" if torch.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_plt() -> None:
    rc = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.monospace": ["Computer Modern Typewriter", "DejaVu Sans Mono"],
    }
    plt.rcParams.update(rc)
