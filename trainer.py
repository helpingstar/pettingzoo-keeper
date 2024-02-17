import yaml
import os
from pikazoo import pikazoo_v0
from pikazoo.wrappers import RewardByBallPosition, RewardInNormalState


class SelfPlay:
    def __init__(self, experiment_name: str) -> None:
        with open("config.yaml") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        self.experiment_name = experiment_name

        self.env_dict = dict()
        for i, agent_type in enumerate(args["agent"]["agent_type"]):
            if agent_type == "normal":
                env = pikazoo_v0.env(render_mode=None)
                self.env_dict[agent_type] = [env]
            elif agent_type == "neg_on_all_state":
                env = pikazoo_v0.env(render_mode=None)
                env = RewardInNormalState(env, args["agent"]["reward"][i])
                self.env_dict[agent_type] = [env]
            elif agent_type == "pos_on_half":
                env1 = pikazoo_v0.env(render_mode=None)
                env1 = RewardByBallPosition(
                    env1, {1, 4}, args["agent"]["reward"][i], False
                )
                env2 = pikazoo_v0.env(render_mode=None)
                env2 = RewardByBallPosition(
                    env2, {1, 4}, args["agent"]["reward"][i], True
                )
                self.env_dict[agent_type] = [env1, env2]
            elif agent_type == "pos_on_quarter_down":
                env1 = pikazoo_v0.env(render_mode=None)
                env1 = RewardByBallPosition(
                    env1, {4}, args["agent"]["reward"][i], False
                )
                env2 = pikazoo_v0.env(render_mode=None)
                env2 = RewardByBallPosition(env2, {4}, args["agent"]["reward"][i], True)
                self.env_dict[agent_type] = [env1, env2]
            else:  # "pos_on_quarter_up"
                env1 = pikazoo_v0.env(render_mode=None)
                env1 = RewardByBallPosition(
                    env1, {1}, args["agent"]["reward"][i], False
                )
                env2 = pikazoo_v0.env(render_mode=None)
                env2 = RewardByBallPosition(env2, {1}, args["agent"]["reward"][i], True)
                self.env_dict[agent_type] = [env1, env2]

        # Create a experiment folder
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

            for agent_type in args["agent"]["agent_type"]:
                agent_folder_path = os.path.join(self.experiment_name, agent_type)
                os.makedirs(agent_folder_path)
                os.makedirs(os.path.join(agent_folder_path, "independent"))
                os.makedirs(os.path.join(agent_folder_path, "shared"))


if __name__ == "__main__":
    selfplay = SelfPlay("test1")

    print()
