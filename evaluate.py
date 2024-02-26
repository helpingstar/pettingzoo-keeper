import numpy as np
import torch
from pikazoo import pikazoo_v0
from network import PPOAgent
from gymnasium.experimental.wrappers import RecordVideoV0
import supersuit as ss
from pikazoo.wrappers import RecordEpisodeStatistics
import utils


class SharedEvaluater:
    def __init__(self, experiment_name) -> None:
        self.experiment_name = experiment_name

        self.rec_env = pikazoo_v0.env(render_mode="rgb_array")
        self.rec_env = RecordVideoV0(self.rec_env)
        self.rec_env = ss.dtype_v0(self.rec_env, np.float32)
        self.rec_env = ss.normalize_obs_v0(self.rec_env)

        self.nml_env = pikazoo_v0.env(render_mode=None)
        self.nml_env = ss.dtype_v0(self.nml_env, np.float32)
        self.nml_env = ss.normalize_obs_v0(self.nml_env)

        self.weights = utils.get_all_agents_weights(self.experiment_name, True, False)

        self.agent1 = PPOAgent()
        self.agent2 = PPOAgent()

    def evaluate_two_agents(self, agent_type1, agent_type2):
        pass

    def fight(
        self, n_episode: int, agent1: PPOAgent, agent2: PPOAgent, record_video: bool
    ):
        assert not (
            agent1.training or agent2.training
        ), "The two networks should be in eval mode."

        winning_count = np.array([0, 0])
        cumulated_score = np.array([0, 0])

        if record_video:
            env = self.rec_env
        else:
            env = self.nml_env
        agents = [agent1, agent2]
        with torch.inference_mode():
            for i in range(n_episode):
                observations, infos = env.reset()
                while env.agents:
                    # this is where you would insert your policy
                    actions = {
                        agent: agents[i].get_action_and_value(observations)[0]
                        for i, agent in enumerate(env.agents)
                    }
                    observations, rewards, terminations, truncations, infos = env.step(
                        actions
                    )
                if env.physics.player1.is_winner:
                    winning_count[0] += 1
                else:
                    winning_count[1] += 1
                cumulated_score[0] += env.scores[0]
                cumulated_score[1] += env.scores[1]
        env.close()
        p1_score_rate = cumulated_score[0] / cumulated_score.sum()
        p1_winning_rate = winning_count[0] / winning_count.sum()

        return p1_score_rate, p1_winning_rate


if __name__ == "__main__":
    evaluater = Evaluater("ex1")
