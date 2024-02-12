import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from pettingzoo.utils import ParallelEnv
from pikazoo import pikazoo_v0


def evaluate(
    n_episode: int,
    env: ParallelEnv = None,
    player1: nn.Module = None,
    player2: nn.Module = None,
):
    winning_count = np.array([0, 0])
    cumulated_score = np.array([0, 0])
    # env = pikazoo_v0.env(render_mode=None)

    # player1.eval()
    # player2.eval()

    # with torch.inference_mode():
    for i in tqdm(range(n_episode)):
        observations, infos = env.reset()
        while env.agents:
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            # actions = {env.possible_agents[0]: player1, env.possible_agents[1]: player2}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        if env.physics.player1.is_winner:
            winning_count[0] += 1
        else:
            winning_count[1] += 1
        cumulated_score[0] += env.scores[0]
        cumulated_score[1] += env.scores[1]
    print(cumulated_score, cumulated_score[0] / cumulated_score.sum())
    print(winning_count, winning_count[0] / winning_count.sum())

    env.close()


if __name__ == "__main__":
    env = pikazoo_v0.env()
    evaluate(1000, env)
