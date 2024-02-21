import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from pettingzoo.utils import ParallelEnv
from pikazoo import pikazoo_v0
from network import PPOAgent


def evaluate_pikazoo(
    n_episode: int,
    agent1: PPOAgent,
    agent2: PPOAgent,
):
    winning_count = np.array([0, 0])
    cumulated_score = np.array([0, 0])
    env = pikazoo_v0.env(render_mode=None)

    agent1.eval()
    agent2.eval()

    with torch.inference_mode():
        for i in range(n_episode):
            observations, infos = env.reset()
            while env.agents:
                # this is where you would insert your policy
                action1 = agent1.get_action_and_value()[0]
                action2 = agent2.get_action_and_value()[0]
                actions = {"player_1": action1, "player_2": action2}
                observations, rewards, terminations, truncations, infos = env.step(
                    actions
                )
            if env.physics.player1.is_winner:
                winning_count[0] += 1
            else:
                winning_count[1] += 1
            cumulated_score[0] += env.scores[0]
            cumulated_score[1] += env.scores[1]
    env.close()
    p1_score_rate = cumulated_score[0] / cumulated_score.sum()
    p1_winning_rate = winning_count[0] / winning_count.sum()

    return p1_score_rate, p1_winning_rate


if __name__ == "__main__":
    env = pikazoo_v0.env()
