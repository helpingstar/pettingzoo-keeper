import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, linear_size=256, n_action=13):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(35, linear_size)),
            nn.ReLU(),
            layer_init(nn.Linear(linear_size, linear_size)),
            nn.ReLU(),
            layer_init(nn.Linear(linear_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(35, linear_size)),
            nn.ReLU(),
            layer_init(nn.Linear(linear_size, linear_size)),
            nn.ReLU(),
            layer_init(nn.Linear(linear_size, n_action), std=0.01),
        )

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
