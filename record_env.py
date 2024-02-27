from pikazoo import pikazoo_v0
from gymnasium.experimental.wrappers import RecordVideoV0
import utils
import torch
from network import PPOAgent
import supersuit as ss
import numpy as np
import os


def record_fight(version=-1):
    weights = utils.get_all_agents_weights("ex1", True, False)
    newest_version = len(list(weights.values())[0])
    if version == -1:
        version = newest_version
    assert 0 < version <= newest_version
    the_newest_weights = dict()
    for k, v in weights.items():
        the_newest_weights[k] = v[version - 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent1 = PPOAgent().to(device)
    agent2 = PPOAgent().to(device)
    iter = 2

    target_path = os.path.join("data", "ranker_fight", f"{version:03d}")

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for k1, v1 in the_newest_weights.items():
        for k2, v2 in the_newest_weights.items():
            print(f"{k1} vs {k2}")
            # load weights
            agent1.load_state_dict(torch.load(v1))
            agent2.load_state_dict(torch.load(v2))

            env = pikazoo_v0.env(render_mode="rgb_array")
            env = ss.dtype_v0(env, np.float32)
            env = ss.normalize_obs_v0(env)
            env = RecordVideoV0(
                env,
                video_folder=target_path,
                episode_trigger=lambda x: True,
                name_prefix=f"{k1}_{k2}",
                disable_logger=True,
            )
            for _ in range(iter):
                (observations, infos) = env.reset()

                observations["player_1"] = torch.Tensor(observations["player_1"]).to(device)
                observations["player_2"] = torch.Tensor(observations["player_2"]).to(device)

                while env.agents:
                    # this is where you would insert your policy
                    actions = {
                        "player_1": agent1.get_action_and_value(observations["player_1"])[0],
                        "player_2": agent2.get_action_and_value(observations["player_2"])[0],
                    }

                    (observations, rewards, terminations, truncations, infos) = env.step(actions)
                    observations["player_1"] = torch.Tensor(observations["player_1"]).to(device)
                    observations["player_2"] = torch.Tensor(observations["player_2"]).to(device)
            env.close()


if __name__ == "__main__":
    record_fight()
