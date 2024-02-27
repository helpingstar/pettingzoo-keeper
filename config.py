experiment_name = "test"
agent = {
    "agent_type": [
        "nml",  # normal
        "nal",  # negative all
        "phf",  # positive half
    ],
    "reward": [None, -0.001, 0.001],
}
selfplay = {"weight_divison": 2}
wandb = {"silent": "true"}
env = {"winning_score": 5}

agent["n_agent_type"] = len(agent["agent_type"])
