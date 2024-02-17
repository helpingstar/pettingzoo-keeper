import torch.nn as nn
import yaml
import os
from pathlib import Path


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
        for i in range(1, 5):
            file_path = Path(directory) / f"saved_{i}.txt"
            file_path.touch()
        file_path = Path(directory) / f"main.txt"
        file_path.touch()


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


def get_all_shared_weights(start_dir):
    weights_list = []

    for root, dirs, files in os.walk(start_dir):
        if "independent" in root.split(os.sep):
            continue
        for file in files:
            if "saved" in file:
                weights_list.append(os.path.join(root, file))

    return weights_list


def get_independent_weights(start_dir, folder_name, file_keyword):
    target_dir = os.path.join(start_dir, folder_name, "independent")
    file_paths = []

    # Check if the target directory exists
    if not os.path.exists(target_dir):
        return file_paths

    for root, dirs, files in os.walk(target_dir):
        # Add the path of files that contain the specified keyword in their name
        for file in files:
            if file_keyword in file:
                file_paths.append(os.path.join(root, file))
    
    return file_paths

delete_files_in_leaf_folders("test1")