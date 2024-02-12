from pikazoo import pikazoo_v0
import supersuit as ss
import pettingzoo

env = pikazoo_v0.env()
env = ss.pettingzoo_env_to_vec_env_v1(env)
envs = ss.concat_vec_envs_v1(env, 4 // 2, num_cpus=0, base_class="gymnasium")

obs, info = envs.reset()

print(obs, info)
