import subprocess
from utils import WeightHandler
import os


arguments = {
    "n_cpus": os.cpu_count(),
    "round": None,
    "player_train": None,
    "path_pool": None,
    "index_infer": None,
    "load_weight_train": None,
    "load_weight_infer": None,
}

pull_command = ["git", "pull", "origin", "main"]

if __name__ == "__main__":
    subprocess.run(pull_command)
    weight_handler = WeightHandler()
    arguments["path_pool"] = weight_handler.path_pool

    while True:
        arguments["player_train"] = weight_handler.get_train_turn()
        arguments["round"] = weight_handler.get_round()
        arguments["load_weight_train"], arguments["load_weight_infer"], arguments["index_infer"] = (
            weight_handler.get_weight_paths_and_idx()
        )
        arguments_list = ["python", "ppo_vec_two_network.py"]
        for k, v in arguments.items():
            arguments_list.append(f"--{k}")
            arguments_list.append(f"{v}")
        subprocess.run(arguments_list)
