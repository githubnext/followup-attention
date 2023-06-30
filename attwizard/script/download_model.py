import yaml

import os
from os.path import join

from pathlib import Path
import wget
import shutil

from attwizard.script.utils import read_config_file

import argparse

if __name__ == "__main__":

    # read the first argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to the config file")
    args = parser.parse_args()
    # read the config file
    config = read_config_file(args.config_file)
    # config = read_config_file("config/download_config.yaml")

    for url_model in config["models_to_download"]:
        print(f"Downloading {url_model} ...")
        repo_name = url_model.split("/")[-1]
        target_folder_name = url_model.replace("/", "_")
        target_folder = join(config["local_model_folder"], target_folder_name)
        # check if target folder existis and it is not empty
        if os.path.exists(target_folder) and os.listdir(target_folder):
            print(f"Folder {target_folder} contains files already. Skipping download.")
            continue
        # create folder if it doesn't exist
        if not os.path.exists(target_folder):
            Path(target_folder).mkdir(parents=True, exist_ok=True)
        # clone the repository
        current_folder = os.getcwd()
        os.chdir(target_folder)
        os.system(f"git clone https://huggingface.co/{url_model}")
        os.chdir(current_folder)
        # remove .git folder
        shutil.rmtree(join(target_folder, repo_name, ".git"))
        path_pytorch_model = join(
            target_folder, repo_name, "pytorch_model.bin")
        # print the size of the model file
        print(
            f"Size of placeholder pytorch_model.bin file (in MB): " +
            f"{os.path.getsize(path_pytorch_model) / 1024 / 1024} MB")
        # ask the user if he wants to replace the placeholder
        # with the model file
        if os.path.exists(path_pytorch_model):
            print(f"Do you want to replace it with the model file?")
            answer = input("[y/n] ")
            if answer == "y":
                # remove placeholder
                os.remove(path_pytorch_model)
                # download the model
                wget.download(
                    f"https://huggingface.co/{url_model}/resolve/main/pytorch_model.bin",
                    join(target_folder, repo_name, "pytorch_model.bin"))
                print(f"Downloaded {url_model}")
            else:
                print("Skipping download.")
    print("Done")
