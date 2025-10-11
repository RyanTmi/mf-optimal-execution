from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    maturity: float
    sigma: float
    gamma: float
    ca: float
    cx: float
    cg: float
    x0_mean: float
    x0_std: float


def get_example_config() -> Config:
    return Config(
        maturity=1.0,
        sigma=0.5,
        gamma=1.75,
        ca=1.0,
        cx=2.0,
        cg=0.3,
        x0_mean=0.5,
        x0_std=0.3,
    )
