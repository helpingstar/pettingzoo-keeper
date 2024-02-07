import tyro
from dataclasses import dataclass
from pettingzoo.utils import AECEnv, ParallelEnv
import torch.nn as nn
from utils import schedule_matches, copy_network


@dataclass
class Args:
    is_parallel = True
    is_symmetry = False
    agent_pool = 4
    cycle = 10
    n_match = 2
    rating = "ELO"  # ELO / WinningRate


class SelfPlay:
    def __init__(
        self,
        env: AECEnv | ParallelEnv,
        is_symmetry: bool,
        n_agent_pool: int,
        cycle: int,
        n_match: int,
        rating: str,
        network: nn.Module,
        algorithm,
    ) -> None:
        assert n_agent_pool % 2 == 0
        assert is_symmetry or n_match % 2 == 0
        assert isinstance(env, AECEnv) or isinstance(env, ParallelEnv)

        self.env = env
        self.is_parallel = True if isinstance(env, ParallelEnv) else False
        self.is_symmetry = is_symmetry
        self.n_agent_pool = n_agent_pool
        self.cycle = cycle
        self.n_match = n_match
        self.rating = rating

        self.schedule = schedule_matches(n_agent_pool)

        self.candidates = [network() for _ in range(n_agent_pool)]
        self.copy_network(0)

    def copy_network(self, cand_idx: int):
        for i in range(self.n_agent_pool):
            if i == cand_idx:
                continue
            copy_network(self.candidates[cand_idx], self.candidates[i])

    def train(self):
        for c in range(self.cycle):
            self.league

            # evaluate
            # get winner
            # copy network

    def league(self):
        for m in range(self.n_match):
            for round in self.schedule:  # round: List[Tuple[int]]
                for c1, c2 in round:
                    self.learn(c1, c2)

    def learn(self, c1, c2):
        c1_network = self.candidates[c1]
        c2_network = self.candidates[c2]

        # battle

    def evaluate(self):
        pass


if __name__ == "__main__":
    pass
