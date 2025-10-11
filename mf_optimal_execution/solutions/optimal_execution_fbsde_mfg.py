from ..config import Config
from ..models import PriceImpactModel

import torch
from scipy.integrate import quad

import math


class OptimalExecutionSolutionFBSDE_MFG:
    def __init__(self, model: PriceImpactModel, config: Config) -> None:
        self.__model = model
        self.__config = config

    def alpha(self, t: float, x: torch.Tensor) -> torch.Tensor:
        m = self.__model

        eta_bar = self.__eta_bar(t)
        eta = self.__eta(t)
        x_bar = self.__x_bar(t)

        return -(eta * x + (eta_bar - eta) * x_bar) / m.ca

    def __eta(self, t: float) -> float:
        m = self.__model

        a = m.ca * math.sqrt(m.cx / m.ca)
        e = math.exp(2 * math.sqrt(m.cx / m.ca) * (self.__config.maturity - t))

        num = a - m.cg - (a + m.cg) * e
        den = a - m.cg + (a + m.cg) * e
        return -a * num / den

    def __eta_bar(self, t: float) -> float:
        m = self.__model

        b = 1 / m.ca
        c = m.cx
        d = -m.gamma / (2 * m.ca)
        r = d**2 + b * c
        dp, dm = -d + math.sqrt(r), -d - math.sqrt(r)

        e = math.exp((dp - dm) * (self.__config.maturity - t))

        num = -c * (e - 1) - m.cg * (dp * e - dm)
        den = (dm * e - dp) - m.cg * b * (e - 1)
        return num / den

    def __x_bar(self, t: float) -> float:
        m = self.__model
        return self.__config.x0_mean * math.exp(-quad(self.__eta_bar, 0, t)[0] / m.ca)
