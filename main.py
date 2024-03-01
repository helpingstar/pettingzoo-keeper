import trainer
import wandb
import os
import config

os.environ["WANDB_SILENT"] = config.wandb["silent"]

if __name__ == "__main__":
    selfplay = trainer.SelfPlay("ex1", True, False)
    selfplay.start(300)
