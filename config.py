experiment_name = "test"
agent = {
    "n_agent_type": 5,
    "agent_type": [
        "normal",
        "neg_on_all_state",
        "pos_on_half",
        "pos_on_quarter_down",
        "pos_on_quarter_up",
    ],
    "reward": [None, -0.001, 0.001, 0.001, 0.001],
}
selfplay = {"weight_divison": 5}
