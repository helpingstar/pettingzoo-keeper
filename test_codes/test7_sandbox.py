from pikazoo import pikazoo_v0
import supersuit as ss
import numpy as np
from tqdm import tqdm
import time

env = pikazoo_v0.env(render_mode=None)
# env = ss.dtype_v0(env, np.float32)
# env = ss.normalize_obs_v0(env)
observations, infos = env.reset()
s1, s2 = env.physics.player1.state, env.physics.player2.state
if s1 > 4 or s2 > 4:
    print(s1, s2)
for i in range(100000):
    # this is where you would insert your policy
    # actions = {agent: 0 for agent in env.agents}
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    s1, s2 = env.physics.player1.state, env.physics.player2.state
    if s1 > 4 or s2 > 4:
        print(s1, s2)
    if not env.agents:
        observations, infos = env.reset()
        s1, s2 = env.physics.player1.state, env.physics.player2.state
        if s1 > 4 or s2 > 4:
            print(s1, s2)
env.close()


test = {"cateogry": {"a": 1, "b": [1, 2, 3, 4], "c": [2, 3]}}
