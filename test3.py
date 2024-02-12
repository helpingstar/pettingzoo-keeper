from pikazoo import pikazoo_v0

env = pikazoo_v0.env()

print(env.observation_space().shape)
print(env.action_space("player_1").sample())
