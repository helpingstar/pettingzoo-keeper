import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, train_type="", agent_type="", weight_type=""):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(36, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(36, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 18), std=0.01),
        )

        # shared / independent
        self._train_type = train_type
        # nml / nal / phf
        self._agent_type = agent_type
        # main / saved
        self._weight_type = weight_type

        print(f"Create PPOAgent {self.net_info()}")

    def net_info(self):
        return f"{self._train_type} {self._agent_type} {self._weight_type}"

    @property
    def train_type(self):
        return self._train_type

    @property
    def agent_type(self):
        return self._agent_type

    @property
    def weight_type(self):
        return self._weight_type

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
