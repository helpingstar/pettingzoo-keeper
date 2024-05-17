from pikazoo import pikazoo_v0
import supersuit as ss
import numpy as np
from pikazoo.wrappers import RecordEpisodeStatistics, NormalizeObservation
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(35, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(35, 18),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


num_envs = 6

env = pikazoo_v0.env(winning_score=2, render_mode=None)
# env = NormalizeObservation(env)
# env = RecordEpisodeStatistics(env)


env = ss.pettingzoo_env_to_vec_env_v1(env)
envs = ss.concat_vec_envs_v1(env, num_envs, num_cpus=6, base_class="gymnasium")

print(envs.idx_starts)
agent = Agent()

p1_indices = np.array(envs.idx_starts[:-1])
p2_indices = p1_indices + 1


def split_p(x):
    return x[p1_indices], x[p2_indices]


# print(p1_indices, p2_indices)

observations_all, infos_all = envs.reset()

print(observations_all.shape)
action = agent.get_action_and_value(torch.Tensor(observations_all))[0]
action_np = action.cpu().numpy()
print()

# print(split_p(observations_all))

# envs.step()

# print(envs.observation_space())
# print(envs.action_space())

# for i in range(100000):
#     actions = np.random.randint(0, 18, size=(num_envs * 2))
#     print(observations.shape)
#     observations, rewards, terminations, truncations, infos = envs.step(actions)
#     print("-----")
#     # print(infos)
# env.close()

# observations, infos = env.reset()
# for i in range(100000):
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     observations, rewards, terminations, truncations, infos = env.step(actions)
#     if not env.agents:
#         observations, infos = env.reset()
# env.close()


envs.close()
