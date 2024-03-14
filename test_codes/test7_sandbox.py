from pikazoo import pikazoo_v0
import supersuit as ss
import numpy as np
from pikazoo.wrappers import RecordEpisodeStatistics, NormalizeObservation

num_envs = 8

env = pikazoo_v0.env(winning_score=2, render_mode=None)
env = NormalizeObservation(env)
env = RecordEpisodeStatistics(env)


env = ss.pettingzoo_env_to_vec_env_v1(env)
envs = ss.concat_vec_envs_v1(env, num_envs, num_cpus=8, base_class="gymnasium")
observations, infos = envs.reset()

print(envs.observation_space())
print(envs.action_space())

for i in range(100000):
    actions = np.random.randint(0, 18, size=(num_envs * 2))
    print(observations.shape)
    observations, rewards, terminations, truncations, infos = envs.step(actions)
    print("-----")
    # print(infos)
env.close()

# observations, infos = env.reset()
# for i in range(100000):
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     observations, rewards, terminations, truncations, infos = env.step(actions)
#     if not env.agents:
#         observations, infos = env.reset()
# env.close()
