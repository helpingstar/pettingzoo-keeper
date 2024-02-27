experiment_name = "test"
agent = {
    "agent_type": ["nml", "nal", "phf"],  # normal, negative all, positive half
    "reward": [None, -0.001, 0.001],
}
selfplay = {"weight_divison": 2}
wandb = {"silent": "true"}
env = {"winning_score": 15}

agent["n_agent_type"] = len(agent["agent_type"])
