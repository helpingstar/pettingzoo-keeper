from pikazoo import pikazoo_v0
from pikazoo.wrappers import ActionConverter
from network import Agent
import torch

env = pikazoo_v0.env(render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

observations, infos = env.reset()

agent_train = Agent().to(device)
agent_no_train = Agent().to(device)

observations["player_1"] = torch.Tensor(observations["player_1"]).to(device)
observations["player_2"] = torch.Tensor(observations["player_2"]).to(device)


while env.agents:
    # this is where you would insert your policy
    actions = {
        "player_1": agent_train.get_action_and_value(observations["player_1"]),
        "player_2": agent_no_train.get_action_and_value(observations["player_2"]),
    }
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    observations["player_1"] = torch.Tensor(observations["player_1"]).to(device)
    observations["player_2"] = torch.Tensor(observations["player_2"]).to(device)
env.close()
