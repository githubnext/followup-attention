import re
import os
import pandas as pd
import time
import json
from typing import Union, List, Dict, Any
from tqdm import tqdm


def get_most_recent(folder: str, prefix: str, extension: str = ".csv"):
    """Get the most recent file conforming to the format `pre_fix_1234.csv`.

    The number refers to the timestamp in seconds when the file was created.
    """
    if extension[0] != ".":
        extension = f".{extension}"
    existing_files = [
        (int(re.search(f"{prefix}_([0-9]+){extension}", f).group(1)), f)
        for f in os.listdir(folder)
        if f.startswith(prefix)]
    # get the latest file
    return sorted(existing_files)[-1][1]


def save_with_timestamp(
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        folder: str, prefix: str, extension: str = ".csv"):
    """Save the dataframe to a file with a timestamp.
    """
    if extension[0] != ".":
        extension = f".{extension}"
    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}{extension}"
    if extension == ".csv":
        data.to_csv(os.path.join(folder, filename), index=False)
    elif extension == ".json":
        with open(os.path.join(folder, filename), "w") as f:
            json.dump(data, f)
    return filename


def save_if_not_existing(
        data: List[Dict[str, Any]],
        output_folder: str):
    """Store records in data as files if they are not yet there."""
    # get existing data, avoid writing them
    existing_data = os.listdir(output_folder)
    print("Existing data: ", len(existing_data))
    data = [
        record for record in data
        if record["local_filename"] + ".json" not in existing_data]
    for record in tqdm(data):
        print(f"Saving {record['local_filename']}.json ...")
        with open(
                os.path.join(
                    output_folder, record["local_filename"] + ".json"), "w") as f:
            json.dump(record, f)
