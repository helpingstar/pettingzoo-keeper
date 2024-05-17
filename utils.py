import numpy as np
from numpy.typing import NDArray
import os
from network import Agent
import torch


class ElementHandler:
    def __init__(self, idx_starts: list[int], train_id) -> None:
        self.num_envs = len(idx_starts)
        if train_id == "p1":
            self.indices_train = np.array(idx_starts)
            self.indices_infer = self.indices_train + 1
        else:
            self.indices_infer = np.array(idx_starts)
            self.indices_train = self.indices_infer + 1

    def get_train(self, element, id=None):
        if id == "info":
            return [element[i] for i in self.indices_train]
        return element[self.indices_train]

    def get_infer(self, element):
        return element[self.indices_infer]

    def split(self, element):
        return element[self.indices_train], element[self.indices_infer]

    def combine(self, element_train: NDArray, element_infer: NDArray):
        element = np.zeros(shape=(self.num_envs * 2,), dtype=element_train.dtype)
        element[self.indices_train] = element_train
        element[self.indices_infer] = element_infer
        return element


class WeightHandler:
    def __init__(self, path_pool="data/weight/pool"):
        self._dir = dict()
        self._players = ("p1", "p2")
        self._dir["pool"] = path_pool
        self._dir["p1"] = os.path.join(path_pool, "p1")
        self._dir["p2"] = os.path.join(path_pool, "p2")
        self._weight = dict()
        self._weight["p1_main"] = os.path.join(self._dir["p1"], "p1_main.pth")
        self._weight["p2_main"] = os.path.join(self._dir["p2"], "p2_main.pth")

        for directory in self._dir.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory {directory} is created.")

        self.n_weight = {p: len(os.listdir(self._dir[p])) for p in self._players}

        if self.n_weight["p1"] == 0 or self.n_weight["p2"] == 0:
            dummy_agent = Agent()
            if self.n_weight["p1"] == 0:
                torch.save(dummy_agent.state_dict(), self._weight["p1_main"])
                torch.save(dummy_agent.state_dict(), self.get_weight_path("p1", 1))
                print(f"Initial weight for p1 is created.")
            if self.n_weight["p2"] == 0:
                torch.save(dummy_agent.state_dict(), self._weight["p2_main"])
                torch.save(dummy_agent.state_dict(), self.get_weight_path("p2", 1))
                print(f"Initial weight for p2 is created.")
            del dummy_agent

    @property
    def path_pool(self):
        return self._dir["pool"]

    def _update_n_weight(self):
        self.n_weight["p1"] = len(os.listdir(self._dir["p1"]))
        self.n_weight["p2"] = len(os.listdir(self._dir["p2"]))

    def get_train_turn(self):
        self._update_n_weight()
        if self.n_weight["p1"] > self.n_weight["p2"]:
            return "p2"
        else:
            return "p1"

    def get_weight_path(self, player, n):
        return os.path.join(self._dir[player], f"{player}_{n:04d}.pth")

    def get_round(self):
        player_turn = self.get_train_turn()
        return self.n_weight[player_turn]


if __name__ == "__main__":
    weight_handler = WeightHandler("data/weight/pool")
