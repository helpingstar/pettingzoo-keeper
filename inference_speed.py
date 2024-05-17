"""
slow : 20FPS / 0.05
medium : 25FPS / 0.04
fast : 30FPS / 0.033333333
"""

from network import Agent
from dataclasses import dataclass
import tyro
import numpy as np
from pikazoo import pikazoo_v0
import torch
import time


@dataclass
class Args:
    outer_iteration: int = 10
    inner_iteration: int = 60


def speed_check(outer_iter, inner_iter, is_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")
    assert (is_cuda and device.type == "cuda") or (not is_cuda and device.type == "cpu")

    env = pikazoo_v0.env()
    high = env.observation_space().high
    low = env.observation_space().low
    agent = Agent().to(device)

    execution_time_table = np.zeros(shape=(outer_iter, inner_iter))

    for o in range(outer_iter):
        for i in range(inner_iter):
            start_time = time.perf_counter()
            obs = env.observation_space().sample()
            obs = (obs - low) / (high - low)
            obs = torch.Tensor(obs).to(device)
            action = agent.get_action(obs)
            action = action.cpu().numpy()
            execution_time = time.perf_counter() - start_time
            execution_time_table[o][i] = execution_time
    print(execution_time_table.mean())
    print(execution_time_table.sum(axis=1).mean())
    return


if __name__ == "__main__":
    args = tyro.cli(Args)
    speed_check(args.outer_iteration, args.inner_iteration, True)
