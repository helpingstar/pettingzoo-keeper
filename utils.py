import numpy as np


class ElementHandler:
    def __init__(self, idx_starts: list[int], train_id) -> None:
        self.num_envs = len(idx_starts)
        if train_id == "p1":
            self.indices_train = np.array(idx_starts)
            self.indices_infer = self.indices_train + 1
        else:
            self.indices_infer = np.array(idx_starts)
            self.indices_train = self.indices_infer + 1

    def get_train(self, element):
        return element[self.indices_train]
    
    def get_infer(self, element)
        return element[self.indices_infer]

    def split(self, element):
        return element[self.indices_train], element[self.indices_infer]

    def combine(self, element_train, element_infer):
        element = np.zeros(shape=(self.num_envs * 2,))
        element[self.indices_train] = element_train
        element[self.indices_infer] = element_infer
        return element
