from ..config import Config
from ..models import PriceImpactModel

import torch
from scipy.integrate import quad

import math


class OptimalExecutionSolutionPDE_MFG:
    def __init__(self, model: PriceImpactModel, config: Config) -> None:
        self.__model = model
        self.__config = config

    def alpha(self, t: float, x: torch.Tensor) -> torch.Tensor:
        m = self.__model
        return -(self.__a(t) * x + self.__b(t)) / m.ca

    def v(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.__a(t) * x**2 / 2 + self.__b(t) * x + self.__c(t)

    def x_dist(self, t: float) -> tuple[float, float]:
        c = self.__config

        phi_0 = self.__phi(t, 0)
        mu = c.x0_mean * phi_0 - quad(lambda u: self.__phi(t, u) * self.__b(u), 0, t)[0] / c.ca
        var = (c.x0_std * phi_0) ** 2 + c.sigma**2 * quad(lambda u: self.__phi(t, u) ** 2, 0, t)[0]
        return mu, var

    def x_pdf(self, t: float, x: torch.Tensor) -> torch.Tensor:
        mu, var = self.x_dist(t)
        return torch.exp(-0.5 * ((x - mu) ** 2 / var)) / math.sqrt(2 * math.pi * var)

    def a_dist(self, t: float) -> tuple[float, float]:
        a = self.__a(t)
        mu_x, var_x = self.x_dist(t)
        mu_a = -(a * mu_x + self.__b(t)) / self.__config.ca
        var_a = var_x * (a / self.__config.ca) ** 2
        return mu_a, var_a

    def a_pdf(self, t: float, x: torch.Tensor) -> torch.Tensor:
        mu, var = self.a_dist(t)
        return torch.exp(-0.5 * ((x - mu) ** 2 / var)) / math.sqrt(2 * math.pi * var)

    def __phi(self, t: float, s: float) -> float:
        return math.exp(-quad(self.__a, s, t)[0] / self.__config.ca)

    def __a(self, t: float) -> float:
        m = self.__model

        a = math.sqrt(m.cx * m.ca)
        r = (m.cg - a) / (m.cg + a)
        e = r * math.exp(2 * math.sqrt(m.cx / m.ca) * (t - self.__config.maturity))

        num = 1 + e
        den = 1 - e
        return a * num / den

    def __b(self, t: float) -> float:
        m = self.__model

        return -(self.__a(t) * self.__e(t) + m.ca * self.__e_t(t))

    def __c(self, t: float) -> float:
        func = lambda t: self.__config.sigma**2 * self.__a(t) / 2 - self.__b(t) ** 2 / (2 * self.__config.ca)
        return quad(func, t, self.__config.maturity)[0]

    def __e(self, t: float) -> float:
        m = self.__model

        r_p = (-m.gamma + math.sqrt(m.gamma**2 + 4 * m.cx * m.ca)) / (2 * m.ca)
        r_m = (-m.gamma - math.sqrt(m.gamma**2 + 4 * m.cx * m.ca)) / (2 * m.ca)

        a = m.cg + m.ca * r_m
        b = m.cg + m.ca * r_p

        num = a * math.exp(r_p * (t - self.__config.maturity)) - b * math.exp(r_m * (t - self.__config.maturity))
        dem = a * math.exp(-r_p * self.__config.maturity) - b * math.exp(-r_m * self.__config.maturity)
        return self.__config.x0_mean * num / dem

    def __e_t(self, t: float) -> float:
        m = self.__model

        r_p = (-m.gamma + math.sqrt(m.gamma**2 + 4 * m.cx * m.ca)) / (2 * m.ca)
        r_m = (-m.gamma - math.sqrt(m.gamma**2 + 4 * m.cx * m.ca)) / (2 * m.ca)

        a = m.cg + m.ca * r_m
        b = m.cg + m.ca * r_p

        num = a * r_p * math.exp(r_p * (t - self.__config.maturity)) - b * r_m * math.exp(r_m * (t - self.__config.maturity))
        dem = a * math.exp(-r_p * self.__config.maturity) - b * math.exp(-r_m * self.__config.maturity)
        return self.__config.x0_mean * num / dem
