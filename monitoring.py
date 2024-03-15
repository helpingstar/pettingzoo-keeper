from pikazoo import pikazoo_v0
from pikazoo.wrappers import NormalizeObservation, RecordEpisodeStatistics
from gymnasium.experimental.wrappers import RecordVideoV0
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from typing import Dict
import numpy as np
import gymnasium as gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(35, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(35, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 18), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def obs_to_torch(obs: Dict[str, np.ndarray], divide: bool):
    p1_obs = torch.Tensor(obs["player_1"]).to(device)
    p2_obs = torch.Tensor(obs["player_2"]).to(device)
    if divide:
        return p1_obs, p2_obs
    else:
        return torch.stack([p1_obs, p2_obs])


def torch_to_action(arr):
    action_arr = arr.cpu().numpy()
    actions = {"player_1": action_arr[0].item(), "player_2": action_arr[1].item()}
    return actions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
n_episode = 2
weight_path = "runs/pika-zoo__ppo_vec_single__1__1710417091/cleanrl_ppo_vec_single_73240.pt"
is_player1_computer = False
is_player2_computer = True
winning_score = 15

agent = Agent().to(device)
agent.load_state_dict(torch.load(weight_path))

env = pikazoo_v0.env(
    winning_score=winning_score,
    render_mode="rgb_array",
    is_player1_computer=is_player1_computer,
    is_player2_computer=is_player2_computer,
)

# env = RecordVideoV0(env, ".", step_trigger=lambda x: x % 10000 == 0, video_length=10000, fps=60)
env = RecordVideoV0(env, ".", episode_trigger=lambda x: True, fps=60)
env = NormalizeObservation(env)

with torch.inference_mode():
    for i in range(n_episode):
        observations, infos = env.reset()
        while env.agents:
            observations = obs_to_torch(observations, divide=False)
            actions = agent.get_action_and_value(observations)[0]
            actions = torch_to_action(actions)
            observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
