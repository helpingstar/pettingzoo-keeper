import wandb
import config
import utils
import numpy as np
import os

os.environ["WANDB_SILENT"] = "true"

for cycle in range(10):
    schedule = utils.get_schedule(5, 10, 5)
    for agent_train_idx, agent_train in enumerate(config.agent["agent_type"]):
        print(f"{cycle=}, {agent_train_idx=}")
        wandb_config = {"agent_train": agent_train, "cycle": cycle}
        wandb.init(
            project="wandb_test",
            name=f"{cycle:02d}_{agent_train}",
            config=wandb_config,
            mode="offline",
        )
        for div_idx in range(config.selfplay["weight_divison"]):
            for agent_opp_idx, agent_opp in enumerate(config.agent["agent_type"]):
                for is_right, side in enumerate(["left", "right"]):
                    sub_name = f"{div_idx * 5 + agent_opp_idx}_{agent_opp}_{schedule[agent_opp_idx][div_idx]:02d}_{side}"
                    for i in range(100):
                        log = {
                            f"custom_step": i,
                            f"charts/winning_rate/{sub_name}": i
                            + np.random.normal(scale=10.0),
                            f"losses/loss/{sub_name}": 100
                            - i
                            + +np.random.normal(scale=10.0),
                        }
                        wandb.log(log)
        wandb.finish()
