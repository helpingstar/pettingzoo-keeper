import numpy as np
import torch
from pikazoo import pikazoo_v0
from network import PPOAgent
from gymnasium.experimental.wrappers import RecordVideoV0
import supersuit as ss
import utils
import config
import os
from time import time


class SharedEvaluater:
    def __init__(self, experiment_name: str) -> None:
        assert torch.cuda.is_available(), "CUDA IS NOT AVAILABLE"
        self.device = "cuda:0"
        self.experiment_name = experiment_name
        self.dir_name = os.path.join("evaluate", self.experiment_name)
        all_weights = utils.get_all_agents_weights(self.experiment_name, True, False)

        self.n_weight = len(all_weights["nml"])

        self.agents = {}
        for agent_type, weights in all_weights.items():
            self.agents[agent_type] = [
                PPOAgent(f"Evaluater_{agent_type}").to(self.device).eval() for _ in range(len(weights))
            ]

        for agent_type, agents in self.agents.items():
            for i in range(len(agents)):
                utils.load_weights(self.agents[agent_type][i], all_weights[agent_type][i])

        board_shape = (config.agent["n_agent_type"], config.agent["n_agent_type"], self.n_weight, self.n_weight, 2)
        self.score_board = np.zeros(shape=board_shape)
        self.winning_board = np.zeros(shape=board_shape)
        self.agent_to_idx = {agent: i for i, agent in enumerate(config.agent["agent_type"])}

    def evaluate_all(self):
        print("[Evaulating Start]")
        for i in range(config.agent["n_agent_type"]):
            for j in range(config.agent["n_agent_type"]):
                self.evaluate_two_agents(i, j)

        print("[Saving Data]")
        final_dir = os.path.join(self.dir_name, f"{self.n_weight:03d}", "array")
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        np.save(os.path.join(final_dir, f"score_board.npy"), self.score_board)
        np.save(os.path.join(final_dir, f"winning_board.npy"), self.winning_board)

    def evaluate_two_agents(self, agent_idx1, agent_idx2):
        agent_type1 = config.agent["agent_type"][agent_idx1]
        agent_type2 = config.agent["agent_type"][agent_idx2]

        print(f"  [Evaluate {agent_type1} VS {agent_type2}]")

        for i in range(self.n_weight):
            for j in range(self.n_weight):
                print(f"    Fight {agent_type1}_{i} VS {agent_type2}_{j}")
                self.fight(7, agent_type1, agent_type2, i, j, False)

    def fight(
        self,
        n_episode: int,
        agent1_type: str,
        agent2_type: str,
        agent1_weight_idx,
        agent2_weight_idx,
        record_video: bool,
    ):
        agent1 = self.agents[agent1_type][agent1_weight_idx]
        agent2 = self.agents[agent2_type][agent2_weight_idx]

        assert not (agent1.training or agent2.training), "The two networks should be in eval mode."

        winning_count = self.winning_board[self.agent_to_idx[agent1_type]][self.agent_to_idx[agent2_type]][
            agent1_weight_idx
        ][agent2_weight_idx]
        cumulated_score = self.score_board[self.agent_to_idx[agent1_type]][self.agent_to_idx[agent2_type]][
            agent1_weight_idx
        ][agent2_weight_idx]

        if record_video:
            env = pikazoo_v0.env(render_mode="rgb_array")
            env = RecordVideoV0(env)
        else:
            env = pikazoo_v0.env(render_mode=None)

        env = ss.dtype_v0(env, np.float32)
        env = ss.normalize_obs_v0(env)

        agents_net = {env.possible_agents[0]: agent1, env.possible_agents[1]: agent2}
        with torch.inference_mode():
            for i in range(n_episode):
                observations, infos = env.reset()
                while env.agents:
                    # this is where you would insert your policy
                    actions = {
                        agent: agents_net[agent].get_action_and_value(
                            torch.tensor(observations[agent]).to(self.device)
                        )[0]
                        for agent in env.agents
                    }
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                if infos["player_1"]["score"][0] > infos["player_1"]["score"][1]:
                    winning_count[0] += 1
                else:
                    winning_count[1] += 1
                cumulated_score[0] += infos["player_1"]["score"][0]
                cumulated_score[1] += infos["player_1"]["score"][1]
        env.close()


if __name__ == "__main__":
    evaluater = SharedEvaluater("ex1")
    evaluater.evaluate_all()
