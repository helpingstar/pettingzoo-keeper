from pikazoo import pikazoo_v0
from gymnasium.experimental.wrappers import RecordVideoV0

env = pikazoo_v0.env(render_mode="rgb_array", serve="random")
env = RecordVideoV0(env, fps=60, video_folder=".")

observations, infos = env.reset()


while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
