import yaml
import os
from pikazoo import pikazoo_v0
from pikazoo.wrappers import RewardByBallPosition, RewardInNormalState
import config
from network import PPOAgent
import torch
import gc
import utils


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
            agent_type: PPOAgent().to(self.device) for agent_type in self.agent_type
        }
        self.independent_saved_agents = {
            agent_type: PPOAgent().to(self.device) for agent_type in self.agent_type
        }

    def get_env(self):
        self.env_dict = dict()
        for i, agent_type in enumerate(self.agent_type):
            if agent_type == "normal":
                env = pikazoo_v0.env(render_mode=None)
                self.env_dict[agent_type] = [env]
            elif agent_type == "neg_on_all_state":
                env = pikazoo_v0.env(render_mode=None)
                env = RewardInNormalState(env, config.agent["reward"][i])
                self.env_dict[agent_type] = [env]
            elif agent_type == "pos_on_half":
                env1 = pikazoo_v0.env(render_mode=None)
                env1 = RewardByBallPosition(
                    env1, {1, 4}, config.agent["reward"][i], False
                )
                env2 = pikazoo_v0.env(render_mode=None)
                env2 = RewardByBallPosition(
                    env2, {1, 4}, config.agent["reward"][i], True
                )
                self.env_dict[agent_type] = [env1, env2]
            elif agent_type == "pos_on_quarter_down":
                env1 = pikazoo_v0.env(render_mode=None)
                env1 = RewardByBallPosition(env1, {4}, config.agent["reward"][i], False)
                env2 = pikazoo_v0.env(render_mode=None)
                env2 = RewardByBallPosition(env2, {4}, config.agent["reward"][i], True)
                self.env_dict[agent_type] = [env1, env2]
            else:  # "pos_on_quarter_up"
                env1 = pikazoo_v0.env(render_mode=None)
                env1 = RewardByBallPosition(env1, {1}, config.agent["reward"][i], False)
                env2 = pikazoo_v0.env(render_mode=None)
                env2 = RewardByBallPosition(env2, {1}, config.agent["reward"][i], True)
                self.env_dict[agent_type] = [env1, env2]

    def check_number_of_weights(self):
        first_path = os.path.join(self.experiment_name, "normal", "shared")
        self.n_weights = len(os.listdir(first_path))
        for agent in self.agent_type:
            for type in ("independent", "shared"):
                path = os.path.join(self.experiment_name, agent, type)
                assert self.n_weights == len(
                    os.listdir(path)
                ), f"n_weight: {self.n_weights}, {path}: {len(os.listdir(path))}"

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
            torch.save(network.state_dict(), path_independent)
            torch.save(network.state_dict(), path_shared)
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

    def start(self):
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

        for div_idx in range(config.selfplay["weight_divison"]):
            for agent_idx, agent in enumerate(self.agent_type):
                self.shared_saved_agents[agent].load_state_dict(
                    torch.load(
                        shared_saved_weights[agent][schedule[agent_idx][div_idx]]
                    )
                )

                # train self.shared_main_agents[agent] and self.shared_saved_agents[agent]

        # load independent saved
        independent_saved_weights = utils.get_all_agents_weights(
            self.experiment_name, False, False
        )

        # TODO
        # train independent
        # evaluate
        # save weight


if __name__ == "__main__":
    selfplay = SelfPlay("test1")

    print()
