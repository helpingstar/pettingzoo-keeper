import numpy as np
import config
import wandb


class Logging:
    def __init__(self) -> None:
        pass

    def log(self, sub_name):
        for i in range(100):
            log = {
                f"custom_step": i,
                f"charts/winning_rate/{sub_name}": i + np.random.normal(scale=10.0),
                f"losses/loss/{sub_name}": 100 - i + +np.random.normal(scale=10.0),
            }
            wandb.log(log)
