from pikazoo import pikazoo_v0
from pikazoo.wrappers import NormalizeObservation, RecordEpisodeStatistics
from gymnasium.experimental.wrappers import RecordVideoV0
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from typing import Dict
import numpy as np
import gymnasium as gym
import onnxruntime


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(35, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(35, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
        )

    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action


onnx_path = "pika.onnx"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
n_episode = 2
weight_path = "runs/pika-zoo__ppo_vec_single__1__1710496498/cleanrl_ppo_vec_single_152580.pt"
is_player1_computer = False
is_player2_computer = True
winning_score = 15

ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

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

player_idx = int(is_player1_computer)
player = env.possible_agents[player_idx]
player_o = env.possible_agents[player_idx ^ 1]

for i in range(n_episode):
    observations, infos = env.reset()
    while env.agents:
        observations[player] = observations[player].astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: observations[player]}
        ort_outs = ort_session.run(None, ort_inputs)
        actions = {
            player: ort_outs[0],
            player_o: 0,
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
