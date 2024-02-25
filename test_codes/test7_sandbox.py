from pikazoo import pikazoo_v0
import supersuit as ss
import numpy as np
from tqdm import tqdm
import time

env = pikazoo_v0.env(
    is_player1_computer=True, is_player2_computer=True, render_mode="human"
)
# env = ss.dtype_v0(env, np.float32)
# env = ss.normalize_obs_v0(env)
observations, infos = env.reset()
for i in range(100000):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    s1, s2 = env.physics.player1.state, env.physics.player2.state
    if not env.agents:
        observations, infos = env.reset()
env.close()
