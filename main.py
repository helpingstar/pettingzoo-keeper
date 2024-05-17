import subprocess
from utils import WeightHandler

arguments = {
    "n_cpus": 32,
    "round": "",
    "train_player": "",
    "load_weight_train": "",
    "load_weight_infer": "",
    "path_pool": "",
}

if __name__ == "__main__":
    weight_handler = WeightHandler()
    arguments["train_player"] = weight_handler.get_train_turn()
    arguments["round"] = weight_handler.get_round()
    arguments["path_pool"] = weight_handler.path_pool
    # TODO weight_path
    print(weight_handler.get_train_turn())
