import os
from os.path import join

from typing import List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
import time
import torch
import click
import yaml

from attwizard.script.utils_model import run_model_single_instance
from attwizard.script.utils_model import load_tokenizer_and_model
from attwizard.script.utils import read_config_file
from attwizard.script.utils import get_file_content

from attwizard.decoder import extract_att_matrix


def get_files_in_folder(folder_path: str):
    """Read all the files in the given folder.

    Note that this excludes json files.
    """
    files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(join(folder_path, file)):
            files.append(file)
    # exclude all the json files
    return [file for file in files if not file.endswith(".json")]


def truncate_file_content(file_content: str, stop_line_signal: str):
    """Keep all the lines until you find one matching the stop signal."""
    lines = file_content.split("\n")
    for i, line in enumerate(lines):
        if stop_line_signal in line:
            return "\n".join(lines[:i + 1])
    return file_content


def create_prediction(
        config: Dict[str, Any], model_name: str,
        files: List[str], output_folder: str):
    """Create the prediction for the given model over all the files."""
    model, tokenizer = load_tokenizer_and_model(
        config, model_name=model_name)
    escaped_model_name = model_name.replace("/", "_")
    # make sure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(sorted(files))):
        content = get_file_content(join(config["input_data_folder"], file))
        stop_signal_line = config["stop_signal_line"]
        if stop_signal_line is not None:
            content = truncate_file_content(content, stop_signal_line)
        # get prediction data
        config_gen = config["generation"]

        # run multiple predictions
        for i_try in range(config_gen["n_queries"]):
            torch.cuda.empty_cache()
            i_base_filename = file + f"_{i_try}_{escaped_model_name}"
            # measure time taken
            start_time = time.time()

            model_output, metadata = run_model_single_instance(
                model, tokenizer, content,
                n_new_tokens=config_gen["number_of_tokens"],
                max_time_in_seconds=config_gen["max_time_in_seconds"],
                temperature=config_gen["temperature"],
                seed=i_try)
            end_time = time.time()
            generation_time_seconds = end_time - start_time
            print(f"Time taken to predict {file}: " +
                  f"{generation_time_seconds} seconds")
            # get attention data
            config_att = config["attention"]
            att_matrix = extract_att_matrix(
                model_output,
                condense_on_head_dim=config_att["strategy_reduce_head"],
                condense_on_layer_dim=config_att["strategy_reduce_layer"],
                normalization_strategy=config_att["strategy_normalize_tokens"])
            metadata["generation_time_seconds"] = generation_time_seconds
            metadata["source_file"] = file
            metadata["mode_name"] = model_name
            metadata["config_options"] = config
            # make sure that the metadata folder exists
            Path(join(output_folder, "metadata")).mkdir(
                parents=True, exist_ok=True)
            # save the metadata
            with open(join(
                    output_folder, "metadata",
                    f"{i_base_filename}.json"), "w") as f:
                json.dump(metadata, f)
            # save the attention matrix as numpy array
            if config["save_attention_matrix"]:
                Path(join(output_folder, "att_tensor")).mkdir(
                    parents=True, exist_ok=True)
                start_time = time.time()
                print(f"Saving attention matrix for {file}")
                with open(join(
                        output_folder, "att_tensor",
                        f"{i_base_filename}.npy"), 'wb') as f:
                    np.save(f, att_matrix)
                end_time = time.time()
                generation_time_seconds = end_time - start_time
                print(f"Time taken to store attention matrix {file}: " +
                      f"{generation_time_seconds} seconds")
            # save model output
            if config["save_raw_model_output"]:
                torch.save(model_output, join(
                    output_folder, file + f"_{escaped_model_name}.pt"))
            # save generated sequence
            Path(join(output_folder, "generated_sequence")).mkdir(
                parents=True, exist_ok=True)
            with open(join(
                    output_folder, "generated_sequence",
                    f"{i_base_filename}.txt"), "w") as f:
                f.write(metadata["text_generated"])
                f.close()
            # clear the cache in cuda
            torch.cuda.empty_cache()


def process_data(config: Dict[str, Any]):
    """Process all the files in the given folder.

    Note that this skips json files (considered metadata)."""
    folder_path = config["input_data_folder"]
    prompt_files = get_files_in_folder(folder_path)
    for model_short_repo in config["models"]:
        create_prediction(
            config=config, model_name=model_short_repo,
            files=prompt_files, output_folder=config["output_data_folder"])


@click.group()
@click.option(
    '--config', default=None,
    help="Pass path to yaml config file with keys: " +
         "input_data_folder, " +
         "output_data_folder, " +
         "local_model_folder, " +
         "stop_signal_line, " +
         "generation, " +
         "attention, " +
         "models.")
@click.pass_context
def cli(ctx, config):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    # read yaml config file
    if config is not None:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        ctx.obj['CONFIG'] = config


@cli.command()
@click.pass_context
def queryextract(ctx):
    """Query the model for completion and extract attention."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)
    if config:
        click.echo("Using config file information.")
        process_data(config)
    else:
        click.echo("No config file found. Insert one.")


if __name__ == '__main__':
    cli()
