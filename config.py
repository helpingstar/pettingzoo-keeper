experiment_name = "test"
agent = {
    "n_agent_type": 5,
    "agent_type": [
        "normal",
        "ne_all",
        "po_half",
        "po_qd",
        "po_qu",
    ],
    "reward": [None, -0.001, 0.001, 0.001, 0.001],
}
selfplay = {"weight_divison": 5}
