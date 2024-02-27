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
    def __init__(
        self, experiment_name: str, train_shared=True, train_independent=True
    ) -> None:
        assert torch.cuda.is_available(), "CUDA IS NOT AVAILABLE"
        self.device = "cuda:0"

        self.train_shared = train_shared
        self.train_independent = train_independent

        self.experiment_name = experiment_name
        self.agent_type = config.agent["agent_type"]

        if not os.path.exists(self.experiment_name):
            self.init_weights()

        self.check_number_of_weights(self.train_shared, self.train_independent)
        self.get_env()
        if self.train_shared:
            self.shared_main_agents = {
                agent_type: PPOAgent(f"shared {agent_type} main").to(self.device)
                for agent_type in self.agent_type
            }
            self.shared_saved_agents = (
                PPOAgent(f"shared_None_saved").to(self.device).eval()
            )
        if self.train_independent:
            self.independent_main_agents = {
                agent_type: PPOAgent(f"independent_{agent_type}_main").to(self.device)
                for agent_type in self.agent_type
            }
            self.independent_saved_agents = (
                PPOAgent(f"independent_None_saved").to(self.device).eval()
            )

    def get_env(self):
        self.envs: Dict[str, PikaZooPPOStage] = dict()
        for i, agent_type in enumerate(self.agent_type):
            self.envs[agent_type] = PikaZooPPOStage(
                agent_type, config.agent["reward"][i]
            )

    def check_number_of_weights(self, check_shared=True, check_independent=True):
        assert check_independent or check_shared
        type_list = []
        if check_independent:
            type_list.append("independent")
        if check_shared:
            type_list.append("shared")

        first_path = os.path.join(
            self.experiment_name, config.agent["agent_type"][0], "shared"
        )
        self.n_weights = len(os.listdir(first_path))
        for agent in self.agent_type:
            for type in type_list:
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
            network = PPOAgent(f"First Empty Network")
            network = network.to(self.device)

            agent_folder_path = os.path.join(self.experiment_name, agent_type)
            os.makedirs(agent_folder_path)
            if self.train_shared:
                path_shared = os.path.join(agent_folder_path, "shared")
                os.makedirs(path_shared)
                utils.save_weights(network, os.path.join(path_shared, "main.pth"))
                utils.save_weights(network, os.path.join(path_shared, "saved_001.pth"))

            if self.train_independent:
                path_independent = os.path.join(agent_folder_path, "independent")
                os.makedirs(path_independent)
                utils.save_weights(network, os.path.join(path_independent, "main.pth"))
                utils.save_weights(
                    network, os.path.join(path_independent, "saved_001.pth")
                )

            # free GPU model
            network.cpu()
            del network
        gc.collect()
        torch.cuda.empty_cache()

    def load_main_agents(self):
        if self.train_shared:
            shared_main_weights = utils.get_all_agents_weights(
                self.experiment_name, True, True
            )
        if self.train_independent:
            independent_main_weights = utils.get_all_agents_weights(
                self.experiment_name, False, True
            )

        for agent in self.agent_type:
            if self.train_shared:
                utils.load_weights(
                    self.shared_main_agents[agent], shared_main_weights[agent][0]
                )
            if self.train_independent:
                utils.load_weights(
                    self.independent_main_agents[agent],
                    independent_main_weights[agent][0],
                )

    # 1 cycle : train (n_agnet_type * weight_division)
    def cycle1(self):
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
                    utils.load_weights(
                        self.shared_saved_agents,
                        shared_saved_weights[agent_opp][
                            schedule[agent_opp_idx][div_idx]
                        ],
                    )
                    self.envs[agent_train].train(
                        self.shared_main_agents[agent_train],
                        self.shared_saved_agents,
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
                    utils.load_weights(
                        self.independent_saved_agents[agent_opp],
                        independent_saved_weights[agent_train][
                            schedule[agent_opp_idx][div_idx]
                        ],
                    )

                    self.envs[agent_train].train(
                        self.independent_main_agents[agent_train],
                        self.independent_saved_agents[agent_opp],
                        run_name,
                    )
            wandb.finish()

        self.save_weights()
        # evaluate

    # shared vs shared
    def cycle2(self, n):
        print(f"[Start cycle {n:03d}]")
        self.load_main_agents()
        schedule = utils.get_schedule(
            config.agent["n_agent_type"],
            self.n_weights,
            config.selfplay["weight_divison"],
            get_all_weights=True,
        )
        print(f"Schedule: {schedule}")
        # load shared saved(opponent)
        shared_saved_weights = utils.get_all_saved_weights_list(
            self.experiment_name, True
        )
        print(f"Number of weights: {len(shared_saved_weights)}")
        for agent_train_idx, agent_train in enumerate(self.agent_type):
            print(f"  [Train {agent_train}] | cycle: {n}")
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
                sche_idx = schedule[agent_train_idx][div_idx]
                _, agent_opp, _, file_name = utils.get_info_by_path(
                    shared_saved_weights[sche_idx]
                )
                print(
                    f"    [main {agent_train} VS {agent_opp}_{file_name} / {sche_idx}]"
                )
                run_name = f"{div_idx:03d}_{agent_opp}_{sche_idx}"
                utils.load_weights(
                    self.shared_saved_agents, shared_saved_weights[sche_idx], 2
                )
                self.envs[agent_train].train(
                    self.shared_main_agents[agent_train],
                    self.shared_saved_agents,
                    run_name,
                )
            wandb.finish()

        self.save_weights(1)
        # evaluate

    def save_weights(self, tab=0):
        print(f"{' '*(2*tab)}Saving Weights")
        for agent_type in self.agent_type:
            if self.train_shared:
                utils.save_weights(
                    self.shared_main_agents[agent_type],
                    os.path.join(
                        self.experiment_name,
                        agent_type,
                        "shared",
                        f"saved_{self.n_weights+1:03d}.pth",
                    ),
                    tab,
                )
                utils.save_weights(
                    self.shared_main_agents[agent_type],
                    os.path.join(
                        self.experiment_name,
                        agent_type,
                        "shared",
                        "main.pth",
                    ),
                    tab,
                )
            if self.train_independent:
                utils.save_weights(
                    self.independent_main_agents[agent_type],
                    os.path.join(
                        self.experiment_name,
                        agent_type,
                        "independent",
                        f"saved_{self.n_weights+1:03d}.pth",
                    ),
                    tab,
                )
                utils.save_weights(
                    self.independent_main_agents[agent_type],
                    os.path.join(
                        self.experiment_name,
                        agent_type,
                        "independent",
                        "main.pth",
                    ),
                    tab,
                )

    def start(self, n_cycle):
        self.check_number_of_weights(self.train_shared, self.train_independent)
        for i in range(self.n_weights, self.n_weights + n_cycle):
            self.check_number_of_weights(self.train_shared, self.train_independent)
            self.cycle2(i)
