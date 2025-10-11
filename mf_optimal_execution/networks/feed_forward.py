import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_hidden: int,
        activation=torch.tanh,
    ):
        super().__init__()
        self.__act = activation

        self.__input = nn.Linear(input_dim, hidden_dim)
        self.__hiddens = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden - 1)])
        self.__output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__act(self.__input(x))
        for hidden in self.__hiddens:
            x = self.__act(hidden(x))
        return self.__output(x)
