import torch


class PriceImpactModel:
    """
    Price Impact Model for Optimal Execution.

    This class defines the dynamics and cost structure of an optimal
    execution problem under a mean-field setting.
    """

    def __init__(self, sigma: float, gamma: float, ca: float, cx: float, cg: float):
        self.__sigma = sigma
        self.__gamma = gamma
        self.__ca = ca
        self.__cg = cg
        self.__cx = cx

    @property
    def sigma(self) -> float:
        return self.__sigma

    @property
    def gamma(self) -> float:
        return self.__gamma

    @property
    def ca(self) -> float:
        return self.__ca

    @property
    def cg(self) -> float:
        return self.__cg

    @property
    def cx(self) -> float:
        return self.__cx

    def drift(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def diffusion(self) -> float:
        return self.__sigma

    def running_cost(self, x: torch.Tensor, a: torch.Tensor, a_mean: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.__ca * a**2 + 0.5 * self.__cx * x**2 - self.__gamma * x * a_mean

    def terminal_cost(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.__cg * x**2

    # Other methods for derivatives
    def running_cost_dx(self, x: torch.Tensor, a_mean: torch.Tensor) -> torch.Tensor:
        return self.__cx * x - self.__gamma * a_mean

    def terminal_cost_dx(self, x: torch.Tensor) -> torch.Tensor:
        return self.__cg * x

    def hamiltonian_dx(self, x: torch.Tensor) -> torch.Tensor:
        return self.__cx * x
