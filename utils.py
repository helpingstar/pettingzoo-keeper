import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import config
import torch
from network import PPOAgent


def copy_network(src: nn.Module, dest: nn.Module):
    dest.load_state_dict(src.state_dict())


def create_files_in_leaf_folders(directory):
    # 현재 디렉토리가 leaf 디렉토리인지 확인하는 플래그
    is_leaf = True

    # 현재 디렉토리 내의 모든 항목을 반복
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # 항목이 디렉토리인 경우 재귀 호출
        if os.path.isdir(item_path):
            is_leaf = False
            create_files_in_leaf_folders(item_path)

    # 현재 디렉토리가 leaf 디렉토리인 경우
    if is_leaf:
        # 1.txt, 2.txt, 3.txt, 4.txt 파일 생성
        for i in range(1, 19):
            file_path = Path(directory) / f"saved_{i:03d}.txt"
            file_path.touch()
        file_path = Path(directory) / f"main.txt"
        file_path.touch()


def init_dir(experiment_name):
    # Create a experiment folder
    os.makedirs(experiment_name)

    for agent_type in config.agent["agent_type"]:
        agent_folder_path = os.path.join(experiment_name, agent_type)
        os.makedirs(agent_folder_path)
        path_independent = os.path.join(agent_folder_path, "independent")
        path_shared = os.path.join(agent_folder_path, "shared")
        os.makedirs(path_independent)
        os.makedirs(path_shared)


def delete_files_in_leaf_folders(start_path):
    # 현재 폴더가 leaf 폴더인지 확인하기 위해 하위 폴더 존재 여부를 체크합니다.
    is_leaf = True
    for item in os.listdir(start_path):
        if os.path.isdir(os.path.join(start_path, item)):
            # 하위 폴더가 있으면, 현재 폴더는 leaf 폴더가 아니며,
            # 해당 하위 폴더에 대해 재귀적으로 함수를 호출합니다.
            is_leaf = False
            delete_files_in_leaf_folders(os.path.join(start_path, item))

    # 현재 폴더가 leaf 폴더이면, 폴더 내 모든 파일을 삭제합니다.
    if is_leaf:
        for item in os.listdir(start_path):
            item_path = os.path.join(start_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")


def get_all_agents_weights(experiment_name, is_shared, is_main):
    weights_list = dict()
    for agent in config.agent["agent_type"]:
        agent_weight = get_weights(experiment_name, agent, is_shared, is_main)
        weights_list[agent] = agent_weight
    return weights_list


def get_all_saved_weights_list(experiment_name, is_shared):
    result = []
    temp = get_all_agents_weights(experiment_name, is_shared, False)
    for li in temp.values():
        result.extend(li)
    result.sort(key=lambda x: (x[-7:], x))
    return result


def get_weights(experiment_name, agent_name, is_shared, is_main):
    kind_of_train = "shared" if is_shared else "independent"
    kind_of_weight = "main" if is_main else "saved"
    path = os.path.join(experiment_name, agent_name, kind_of_train)
    weight_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if kind_of_weight in file:
                weight_list.append(os.path.join(root, file))
    weight_list.sort()
    return weight_list


def get_schedule(n_agent, n_weights, n_divison, get_all_weights=False):
    schedules = []
    boundaries = []
    if get_all_weights:
        n_weights *= n_agent
    diff = n_weights / n_divison
    for i in range(n_divison):
        boundaries.append(int(i * diff))
    for i in range(n_agent):
        weight_nums = [np.random.randint(d, n_weights) for d in boundaries]
        schedules.append(weight_nums)

    return schedules


def get_run_name(div_idx, agent_opp, agent_opp_idx, schedule):
    return f"{div_idx * 5 + agent_opp_idx:03d}_{agent_opp}_{schedule[agent_opp_idx][div_idx]:03d}"


def get_info_by_path(path: str):
    infos = path.split("/")
    return infos[0], infos[1], infos[2], infos[3]


def load_weights(dst: PPOAgent, src: str, tab: int = 0):
    print(f"{' '*(2*tab)}{dst.name} <- {src}")
    dst.load_state_dict(torch.load(src))


def save_weights(net: PPOAgent, path: str, tab: int = 0):
    print(f"{' '*(2*tab)}{net.name} -> {path}")
    torch.save(net.state_dict(), path)


if __name__ == "__main__":
    # init_dir("ex2")
    # create_files_in_leaf_folders("ex2")
    # paths = get_all_saved_weights_list("ex2", True)
    # print(paths)
    # print(os.path.split(paths[0]))
    print(get_schedule(5, 10, 3, True))
