import gymnasium as gym

env = gym.make("CliffWalking-v0", render_mode=None)
env = gym.wrappers.AutoResetWrapper(env)
observation, info = env.reset(seed=42)
for _ in range(10000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print(observation, info)
        pass

env.close()
