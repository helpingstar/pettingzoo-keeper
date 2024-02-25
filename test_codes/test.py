import gymnasium as gym
from pikazoo import pikazoo_v0
from pikazoo.wrappers import ConvertSingleAgent


def make_env(idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = pikazoo_v0.env(is_player2_computer=True)
            env = ConvertSingleAgent(env, "player_1")
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = pikazoo_v0.env(is_player2_computer=True)
            env = ConvertSingleAgent(env, "player_1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env(i, False, "hi") for i in range(2)],
)

envs.reset()

for _ in range(10000):
    action = envs.action_space.sample()  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = envs.step(action)
    print(infos)
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                print(info)

envs.close()
