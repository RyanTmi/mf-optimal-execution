from ..config import Config
from ..models import PriceImpactModel

import torch
from scipy.integrate import quad

import math


class OptimalExecutionSolutionFBSDE_MFC:
    def __init__(self, model: PriceImpactModel, config: Config) -> None:
        self.__model = model
        self.__config = config

    def alpha(self, t: float, x: torch.Tensor) -> torch.Tensor:
        m = self.__model

        eta_bar = self.__eta_bar(t)
        eta = self.__eta(t)
        x_bar = self.__x_bar(t)

        return -(eta * x + (eta_bar - eta - m.gamma) * x_bar) / m.ca

    def __eta(self, t: float) -> float:
        m = self.__model

        a = math.sqrt(m.cx * m.ca)
        r = (m.cg - a) / (m.cg + a)
        e = r * math.exp(2 * math.sqrt(m.cx / m.ca) * (t - self.__config.maturity))

        num = 1 + e
        den = 1 - e
        return a * num / den

    def __eta_bar(self, t: float) -> float:
        m = self.__model

        r = 1 / m.ca
        a = 2 * m.gamma * r
        b = r * (m.gamma**2 * r - m.cx)

        c1 = (-a - math.sqrt(a**2 - 4 * b)) / 2
        c2 = (-a + math.sqrt(a**2 - 4 * b)) / 2

        e = math.exp((c2 - c1) * (self.__config.maturity - t))
        num = (c2 + r * m.cg) * c1 * e - c2 * (c1 + r * m.cg)
        den = (c2 + r * m.cg) * e - (c1 + r * m.cg)
        return -num / (r * den)

    def __x_bar(self, t: float) -> float:
        m = self.__model

        return self.__config.x0_mean * math.exp(-(quad(self.__eta_bar, 0, t)[0] - m.gamma * t) / m.ca)
