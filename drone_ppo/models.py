import torch
import torch.nn as nn


class Actor_NN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.mu = nn.Sequential(
            nn.Linear(64, 2),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.full((2,), 0.0))  # state independent same for all states..

    def forward(self, x):
        out=self.network(x)
        mu = self.mu(out)
        std = torch.exp(self.log_std)
        return mu, std


class Critic_NN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out = self.network(x)
        return out
