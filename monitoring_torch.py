from pikazoo import pikazoo_v0
from pikazoo.wrappers import NormalizeObservation, RecordEpisodeStatistics, SimplifyAction
from gymnasium.experimental.wrappers import RecordVideoV0
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from typing import Dict
import numpy as np
import gymnasium as gym
from network import Agent


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

p1_map = (0, 1, 2, 3, 4, 6, 7, 10, 11, 12, 13, 14, 16)
p2_map = (0, 1, 2, 4, 3, 7, 6, 10, 12, 11, 13, 15, 17)

normal_13_256 = "data/weight/selfplay/v2/normal_13_256.pt"
normal_18_128 = "data/weight/selfplay/v1/normal_18_128.pt"
rv_13_256 = "data/weight/selfplay/v2/rv1_13_256.pt"

weight_path_p1 = normal_18_128
weight_path_p2 = normal_13_256

p1_n_action = 18
p2_n_action = 13

p1_n_linear = 128
p2_n_linear = 256

p1_n_layer = 1
p2_n_layer = 1


is_player1_computer = False
is_player2_computer = False

winning_score = 15

agent_p1 = Agent(p1_n_linear, p1_n_action, p1_n_layer)
agent_p2 = Agent(p2_n_linear, p2_n_action, p2_n_layer)

agent_p1 = agent_p1.eval().to(device)
agent_p2 = agent_p2.eval().to(device)

agent_p1.load_state_dict(torch.load(weight_path_p1))
agent_p2.load_state_dict(torch.load(weight_path_p2))

rendering = None
recording = None

env = pikazoo_v0.env(
    winning_score=winning_score,
    render_mode=rendering,
    is_player1_computer=is_player1_computer,
    is_player2_computer=is_player2_computer,
)

if rendering == "rgb_array":
    if recording == "step":
        env = RecordVideoV0(env, ".", step_trigger=lambda x: x % 10000 == 0, video_length=10000, fps=60)
    elif recording == "episode":
        env = RecordVideoV0(env, ".", episode_trigger=lambda x: True, fps=60)
env = NormalizeObservation(env)

with torch.inference_mode():
    for i in range(n_episode):
        observations, infos = env.reset()
        while env.agents:
            obs_p1, obs_p2 = obs_to_torch(observations, divide=True)
            actions = {
                "player_1": agent_p1.get_action(obs_p1).cpu().item(),
                "player_2": agent_p2.get_action(obs_p2).cpu().item(),
            }
            if p1_n_action == 13:
                actions["player_1"] = p1_map[actions["player_1"]]
            if p2_n_action == 13:
                actions["player_2"] = p2_map[actions["player_2"]]
            observations, rewards, terminations, truncations, infos = env.step(actions)
        print(infos)
env.close()
