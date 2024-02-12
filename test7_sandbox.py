from pikazoo import pikazoo_v0
import supersuit as ss
import numpy as np
from tqdm import tqdm
import time

env = pikazoo_v0.env(render_mode=None)
env = ss.dtype_v0(env, np.float32)
env = ss.normalize_obs_v0(env)
observations, infos = env.reset()
result = 0.
for i in range(20):
    start = time.time()
    for i in range(10000):
        # this is where you would insert your policy
        actions = {agent: 0 for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        if not env.agents:
            env.reset()
    result += time.time() - start
env.close()
print(result / 20)