import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_linear, n_action, n_layer):
        super().__init__()
        # Critic Network
        critic_layers = [nn.Linear(35, n_linear), nn.ReLU()]
        for _ in range(n_layer):
            critic_layers.append(nn.Linear(n_linear, n_linear))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(n_linear, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Actor Network
        actor_layers = [nn.Linear(35, n_linear), nn.ReLU()]
        for _ in range(n_layer):
            actor_layers.append(nn.Linear(n_linear, n_linear))
            actor_layers.append(nn.ReLU())
        actor_layers.append(nn.Linear(n_linear, n_action))
        self.actor = nn.Sequential(*actor_layers)

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
