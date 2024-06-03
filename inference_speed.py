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
    is_cuda: bool = False
    n_linear: int = 256
    n_layer: int = 2
    n_action: int = 18


def speed_check(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.is_cuda else "cpu")
    assert (args.is_cuda and device.type == "cuda") or (not args.is_cuda and device.type == "cpu")

    env = pikazoo_v0.env()
    high = env.observation_space().high
    low = env.observation_space().low
    agent = Agent(n_linear=args.n_linear, n_action=args.n_action, n_layer=args.n_layer).to(device).eval()

    execution_time_table = np.zeros(shape=(args.outer_iteration, args.inner_iteration))
    with torch.inference_mode():
        for o in range(args.outer_iteration):
            for i in range(args.inner_iteration):
                start_time = time.perf_counter()
                obs = env.observation_space().sample()
                obs = (obs - low) / (high - low)
                obs = torch.Tensor(obs).to(device)
                action = agent.get_action(obs)
                action = action.cpu().numpy()
                execution_time = time.perf_counter() - start_time
                execution_time_table[o][i] = execution_time
    one_execution = execution_time_table.mean()
    inner_execution = execution_time_table.sum(axis=1).mean()

    print(f"{one_execution=}")
    print(f"{inner_execution=}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    speed_check(args)
