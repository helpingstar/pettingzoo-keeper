from pikazoo import pikazoo_v0
from pikazoo.wrappers import NormalizeObservation
import supersuit as ss
import pettingzoo

env = pikazoo_v0.env()
env = NormalizeObservation(env)

observations, infos = env.reset(seed=42)

print(observations["player_2"][0])