from pikazoo import pikazoo_v0
import supersuit as ss
import numpy as np
from gymnasium.wrappers import TimeLimit
from pikazoo.wrappers import NormalizeObservation
from tqdm import tqdm


def check_obs(obs):
    for ob in obs.values():
        if np.any(ob < 0) or np.any(ob > 1):
            print(ob)


env = pikazoo_v0.env(render_mode=None)
env = NormalizeObservation(env)
env = ss.agent_indicator_v0(env)

observations, infos = env.reset()
# for i in tqdm(range(1000000)):
for i in range(1000000):
    check_obs(observations)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    if not env.agents:
        observations, infos = env.reset()
env.close()
