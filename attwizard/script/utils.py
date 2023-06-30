import os
from os.path import join
from typing import Dict, List, Tuple, Union, Any
from typing import Callable
import yaml
import json
import re
import pathlib
from multiprocessing import Pool
import numpy as np


def load_txt_file(filename):
    with open(filename, 'r') as f:
        return f.read()


def read_data_in_parallel(
        base_folder: str,
        file_type_extension: str,
        read_function: Callable = None,
        name_extractor_function: Callable = None,
        n_processes: int = 4,
        extra_filter_regex: str = None) -> List[Any]:
    """Read the data in parallel.

    If you pass a file_type_extension equal to "json", the json library will
    be used to read the files (json.load), whereas if you pass "npy", the numpy
    library will be used (np.load).
    """
    if not file_type_extension[0] == '.':
        file_type_extension = '.' + file_type_extension
    files_to_read = [
        os.path.join(base_folder, f)
        for f in os.listdir(base_folder)
        if f.endswith(file_type_extension)
    ]
    if extra_filter_regex is not None:
        files_to_read = [
            f for f in files_to_read
            if re.search(extra_filter_regex, f)]
    if not read_function:
        if file_type_extension == ".json":
            read_function = load_json_file
        elif file_type_extension == ".npy":
            read_function = np.load
    with Pool(processes=n_processes) as pool:
        data = pool.map(read_function, files_to_read)
        print("n of data read: ", len(data))

        def get_stem(filename):
            return pathlib.Path(os.path.basename(filename)).stem

        if not name_extractor_function:
            name_extractor_function = get_stem
        base_names = [
            name_extractor_function(f)
            for f in files_to_read]
        all_data_dict = {
            k: v for k, v in zip(base_names, data)
        }
    if len(all_data_dict.keys()) == 0:
        print('WARNING: No data found.')
    return all_data_dict


def load_json_file(filename):
    """Load a json file."""
    with open(filename, "r") as f:
        return json.load(f)


def write_json_file(filename, data):
    """Write a json file."""
    with open(filename, "w") as f:
        json.dump(data, f)


def read_config_file(path: str):
    """Read the config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_folder_path(model_folder: str, hugging_short_repo: str):
    """
    Get the local folder for the given repo.
    """
    return join(
        model_folder,
        hugging_short_repo.replace("/", "_"),
        hugging_short_repo.split("/")[-1])


def query_which_model(model_list: List[str]):
    """
    Ask the user to choose a model from the list.
    """
    print("Choose a model:")
    for i, model in enumerate(model_list):
        print(f"{i}: {model}")
    answer = input("[0-{}] ".format(len(model_list) - 1))
    return model_list[int(answer)]


def get_prompt():
    """
    Ask the user to enter a prompt.
    """
    print("Enter a prompt:")
    return input("")


def get_file_content(filepath: str):
    """Get the content of a file."""
    with open(filepath, "r") as f:
        return f.read()
