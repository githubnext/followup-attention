"""This function rewrites all the filenames using the anonymization mapping."""

from distutils.command.config import config
import os
import yaml
from typing import Dict


def anonymize(
        root_folder: str,
        anonymization_dict_file_path: Dict[str, str]):
    """This function rewrites filenames using the anonymization mapping."""
    with open(anonymization_dict_file_path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        for root, dirs, files in os.walk(root_folder):
            # rename folders
            for dir in dirs:
                if key in dir:
                    new_dir_name = dir.replace(key, value)
                    old_dir_name = os.path.join(root, dir)
                    print(f"Renaming FOLDER: {old_dir_name}...")
                    os.rename(old_dir_name, os.path.join(root, new_dir_name))
        for root, dirs, files in os.walk(root_folder):
            # rename files
            for file in files:
                if key in file:
                    new_file_name = file.replace(key, value)
                    old_file_name = os.path.join(root, file)
                    print(f"Renaming FILE: {old_file_name}...")
                    os.rename(old_file_name, os.path.join(root, new_file_name))


if __name__ == "__main__":
    anonymize(
        root_folder=".",
        anonymization_dict_file_path="anonymization_mapping.yaml")
    print("Anonymization complete.")
