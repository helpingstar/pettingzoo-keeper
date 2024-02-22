import yaml
import os
from pikazoo import pikazoo_v0
from pikazoo.wrappers import RewardByBallPosition, RewardInNormalState
import config
from network import PPOAgent
import torch
import gc
import utils
from pikazoo_ppo_stage import PikaZooPPOStage
from typing import List, Dict
import wandb


class SelfPlay:
    def __init__(self, experiment_name: str) -> None:
        assert torch.cuda.is_available(), "CUDA IS NOT AVAILABLE"
        self.device = "cuda:0"

        self.experiment_name = experiment_name
        self.agent_type = config.agent["agent_type"]

        if not os.path.exists(self.experiment_name):
            self.init_weights()

        self.check_number_of_weights()
        self.get_env()
        self.shared_main_agents = {
            agent_type: PPOAgent().to(self.device) for agent_type in self.agent_type
        }
        self.independent_main_agents = {
            agent_type: PPOAgent().to(self.device) for agent_type in self.agent_type
        }
        self.shared_saved_agents = {
            agent_type: PPOAgent().to(self.device).eval()
            for agent_type in self.agent_type
        }
        self.independent_saved_agents = {
            agent_type: PPOAgent().to(self.device).eval()
            for agent_type in self.agent_type
        }

    def get_env(self):
        self.envs: Dict[str, PikaZooPPOStage] = dict()
        for i, agent_type in enumerate(self.agent_type):
            self.envs[agent_type] = PikaZooPPOStage(
                agent_type, config.agent["reward"][i]
            )

    def check_number_of_weights(self):
        first_path = os.path.join(
            self.experiment_name, config.agent["agent_type"][0], "shared"
        )
        self.n_weights = len(os.listdir(first_path))
        for agent in self.agent_type:
            for type in ("independent", "shared"):
                path = os.path.join(self.experiment_name, agent, type)
                assert self.n_weights == len(
                    os.listdir(path)
                ), f"n_weight: {self.n_weights}, {path}: {len(os.listdir(path))}"
        # Subtract 1 from the number of files because self.n_weight is calculated excluding main.pth.
        self.n_weights -= 1

    def init_weights(self):
        # Create a experiment folder
        os.makedirs(self.experiment_name)

        for agent_type in self.agent_type:
            agent_folder_path = os.path.join(self.experiment_name, agent_type)
            os.makedirs(agent_folder_path)
            path_independent = os.path.join(agent_folder_path, "independent")
            path_shared = os.path.join(agent_folder_path, "shared")
            os.makedirs(path_independent)
            os.makedirs(path_shared)

            network = PPOAgent()
            network = network.to(self.device)
            torch.save(network.state_dict(), os.path.join(path_independent, "main.pth"))
            torch.save(
                network.state_dict(), os.path.join(path_independent, "saved_001.pth")
            )
            torch.save(network.state_dict(), os.path.join(path_shared, "main.pth"))
            torch.save(network.state_dict(), os.path.join(path_shared, "saved_001.pth"))
            # free GPU model
            network.cpu()
            del network
        gc.collect()
        torch.cuda.empty_cache()

    def load_main_agents(self):
        shared_main_weights = utils.get_all_agents_weights(
            self.experiment_name, True, True
        )
        independent_main_weights = utils.get_all_agents_weights(
            self.experiment_name, False, True
        )

        for agent in self.agent_type:
            self.shared_main_agents[agent].load_state_dict(
                torch.load(shared_main_weights[agent][0])
            )
            self.independent_main_agents[agent].load_state_dict(
                torch.load(independent_main_weights[agent][0])
            )

    def cycle(self):
        self.load_main_agents()
        schedule = utils.get_schedule(
            config.agent["n_agent_type"],
            self.n_weights,
            config.selfplay["weight_divison"],
        )
        # load shared saved(opponent)
        shared_saved_weights = utils.get_all_agents_weights(
            self.experiment_name, True, False
        )
        # load independent saved
        independent_saved_weights = utils.get_all_agents_weights(
            self.experiment_name, False, False
        )
        print(f"Start train [shared]")
        for agent_train_idx, agent_train in enumerate(self.agent_type):
            # wandb logging
            wandb_config = {
                "agent_train": agent_train,
                "cycle": self.n_weights,
                "train": "shared",
            }
            wandb.init(
                project=self.experiment_name,
                name=f"{self.n_weights:03d}_{agent_train}_sh",
                config=wandb_config,
            )
            for div_idx in range(config.selfplay["weight_divison"]):
                for agent_opp_idx, agent_opp in enumerate(self.agent_type):
                    run_name = utils.get_run_name(
                        div_idx, agent_opp, agent_opp_idx, schedule
                    )
                    self.shared_saved_agents[agent_opp].load_state_dict(
                        torch.load(
                            shared_saved_weights[agent_opp][
                                schedule[agent_opp_idx][div_idx]
                            ]
                        )
                    )
                    self.envs[agent_train].train(
                        self.shared_main_agents[agent_train],
                        self.shared_saved_agents[agent_opp],
                        run_name,
                    )
            wandb.finish()
        print(f"Start train [independent]")
        for agent_train_idx, agent_train in enumerate(self.agent_type):
            # wandb logging
            wandb_config = {
                "agent_train": agent_train,
                "cycle": self.n_weights,
                "train": "independent",
            }
            wandb.init(
                project=self.experiment_name,
                name=f"{self.n_weights:03d}_{agent_train}_idp",
                config=wandb_config,
            )
            for div_idx in range(config.selfplay["weight_divison"]):
                for agent_opp_idx, agent_opp in enumerate(self.agent_type):
                    run_name = utils.get_run_name(
                        div_idx, agent_opp, agent_opp_idx, schedule
                    )
                    self.independent_saved_agents[agent_opp].load_state_dict(
                        torch.load(
                            independent_saved_weights[agent_train][
                                schedule[agent_opp_idx][div_idx]
                            ]
                        )
                    )

                    self.envs[agent_train].train(
                        self.independent_main_agents[agent_train],
                        self.independent_saved_agents[agent_opp],
                        run_name,
                    )
            wandb.finish()

        self.save_weights()
        # evaluate

    def save_weights(self):
        for agent_type in self.agent_type:
            torch.save(
                self.shared_main_agents[agent_type].state_dict(),
                os.path.join(
                    self.experiment_name,
                    agent_type,
                    "shared",
                    f"saved_{self.n_weights+1:03d}",
                ),
            )
            torch.save(
                self.shared_main_agents[agent_type].state_dict(),
                os.path.join(
                    self.experiment_name,
                    agent_type,
                    "shared",
                    f"main",
                ),
            )
            torch.save(
                self.shared_main_agents[agent_type].state_dict(),
                os.path.join(
                    self.experiment_name,
                    agent_type,
                    "independent",
                    f"saved_{self.n_weights+1:03d}",
                ),
            )
            torch.save(
                self.shared_main_agents[agent_type].state_dict(),
                os.path.join(
                    self.experiment_name,
                    agent_type,
                    "independent",
                    f"main",
                ),
            )

    def start(self, n_cycle):
        for i in range(n_cycle):
            print(f"Start cycle {i:03d}")
            self.check_number_of_weights()
            self.cycle()
