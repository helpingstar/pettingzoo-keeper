experiment_name = "test"
agent = {
    "n_agent_type": 5,
    "agent_type": [
        "nml",  # normal
        "nal",  # negative all
        "phf",  # positive half
    ],
    "reward": [None, -0.001, 0.001],
}
selfplay = {"weight_divison": 3}
wandb = {"silent": "true"}
env = {"winning_score": 15}
