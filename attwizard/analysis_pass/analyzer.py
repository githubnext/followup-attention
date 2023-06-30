"""Compute the temporal/followup attention."""

from importlib.metadata import metadata
import os
from os.path import join

from typing import Callable, List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
import time
import torch
import click
import yaml
import pandas as pd

from attwizard.attention_postprocessing import compute_followup_attention
from attwizard.attention_postprocessing import compute_followup_attention_all_layers
from attwizard.attention_postprocessing import max_10_predictions
from attwizard.attention_postprocessing import max_50_predictions
from attwizard.attention_postprocessing import compute_followup_attention_scaled
from attwizard.attention_postprocessing import compute_naive_max_aggregation
from attwizard.attention_postprocessing import compute_naive_mean_aggregation
from attwizard.attention_postprocessing import compute_mean_of_followers
from attwizard.attention_postprocessing import compute_attention_flow_from_tensor
from attwizard.attention_postprocessing import compute_attention_rollout_from_tensor
from attwizard.attention_postprocessing import compute_transitive_attention
from attwizard.attention_postprocessing import generate_uniform_attention
from attwizard.attention_postprocessing import generate_uniform_attention_from_metadata
from attwizard.attention_postprocessing import generate_copy_cat_attention
from attwizard.attention_postprocessing import extract_single_slice
from attwizard.attention_postprocessing import extract_half_and_sum
from attwizard.attention_postprocessing import sum_over_1st_dimension
from attwizard.attention_postprocessing import raw_weights_last_layer
from attwizard.attention_postprocessing import raw_weights_first_layer
from attwizard.attention_postprocessing import generate_gaussian_attention_in_neighborhood_from_metadata
from attwizard.attention_postprocessing import make_symmetric

from attwizard.script.utils import read_data_in_parallel
from attwizard.script.utils import load_json_file, load_txt_file


from multiprocessing import Pool
from itertools import repeat


def derive_attention(args):
    """Derive attention for the given npy file."""
    filename, data_path, metadata_path, output_folder, kwargs, derivation_function = args
    if filename.endswith(".npy"):
        # load the npy file
        input_unit = np.load(join(data_path, filename))
        out_filename = filename
        if 'machine_metadata' in kwargs:
            kwargs['machine_metadata'] = load_json_file(
                join(metadata_path, filename.replace(".npy", ".json")))
    elif filename.endswith(".json"):
        # load the json file
        input_unit = load_json_file(join(data_path, filename))
        out_filename = filename.replace(".json", ".npy")
    followup_matrix = derivation_function(input_unit, **kwargs)
    # convert to numpy if not already
    if not isinstance(followup_matrix, np.ndarray):
        followup_matrix = followup_matrix.numpy()
    # save the followup matrix as numpy array
    np.save(join(output_folder, out_filename), followup_matrix)


def derive_for_experiment(
        config: Dict[str, Any],
        procedure_name: str,
        input_folder: str,
        metadata_folder: str,
        derivation_function: Callable,
        kwargs: Dict[str, Any] = None) -> None:
    """Derive new attention artifacts for the files in the given folder.

    Note that this ignores json files (containing metadata)."""
    if not kwargs:
        kwargs = {}

    data_path = input_folder
    Path(data_path).mkdir(parents=True, exist_ok=True)

    metadata_path = metadata_folder

    output_folder = join(config["analysis_data_folder"], procedure_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # infer the type of the filetypes
    all_files = [
        f for f in os.listdir(data_path)
        if os.path.isfile(join(data_path, f))]
    filetypes = [f.split(".")[-1] for f in all_files]
    if len(filetypes) == 0:
        raise ValueError(
            "No files in the input folder." +
            "Please check your config file (yaml): " +
            "analysis_passes are computed in order, thus any element should " +
            "have either att_tensor as `input_folder` or a name of a " +
            "previous analysis_pass as `input_folder`."
        )
    target_filetype = filetypes[0]
    assert all(f == target_filetype for f in filetypes), \
        "All files must have the same filetype."
    # get all npy files in the folder
    target_files = [
        f for f in os.listdir(data_path) if f.endswith(target_filetype)]
    # sort
    target_files = sorted(target_files)
    target_files = target_files[:]
    # get already existing files
    existing_files = [
        f for f in os.listdir(output_folder) if f.endswith(target_filetype)]
    print(f"Found {len(target_files)} files to process.")
    print(f"Found {len(existing_files)} files already processed. Skippign them.")
    for f in existing_files:
        print(f"Removing {f} from the list of files to process.")
    # remove already existing files
    target_files = [
        f for f in target_files if f not in existing_files]
    print(f"Processing {len(target_files)} files.")

    with Pool(processes=1) as pool:
        args = list(zip(
            target_files,
            repeat(data_path),
            repeat(metadata_path),
            repeat(output_folder),
            repeat(kwargs),
            repeat(derivation_function)))
        print("Starting to derive attention for {} files...".format(len(target_files)))
        print(f"Args: {args}")
        transformed_data = list(tqdm(
            pool.imap(derive_attention, args),
            total=len(target_files)))
        print("n of data transformed: ", len(transformed_data))


@click.group()
@click.option(
    '--config', default=None,
    help="Pass path to yaml config file with keys: " +
         "analysis_data_folder, " +
         "output_data_folder.")
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
def deriveall(ctx):
    """Process extract attention for each analysis pass."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)
    if config:
        click.echo("Using config file information.")
        for analysis_pass in config['analysis_passes']:
            pass_name = analysis_pass['name']
            print(f"Processing analysis pass: {pass_name}")
            # ALL POSSIBLE ROUTINES
            if analysis_pass["function_name"] == 'compute_followup_attention':
                fn = compute_followup_attention
            elif analysis_pass["function_name"] == 'compute_followup_attention_all_layers':
                fn = compute_followup_attention_all_layers
            elif analysis_pass["function_name"] == 'max_10_predictions':
                fn = max_10_predictions
            elif analysis_pass["function_name"] == 'max_50_predictions':
                fn = max_50_predictions
            elif analysis_pass["function_name"] == 'compute_followup_attention_scaled':
                fn = compute_followup_attention_scaled
            elif analysis_pass["function_name"] == 'compute_naive_max_aggregation':
                fn = compute_naive_max_aggregation
            elif analysis_pass["function_name"] == 'compute_naive_mean_aggregation':
                fn = compute_naive_mean_aggregation
            elif analysis_pass["function_name"] == 'compute_mean_of_followers':
                fn = compute_mean_of_followers
            elif analysis_pass["function_name"] == 'compute_attention_flow_from_tensor':
                fn = compute_attention_flow_from_tensor
            elif analysis_pass["function_name"] == 'compute_attention_rollout_from_tensor':
                fn = compute_attention_rollout_from_tensor
            elif analysis_pass["function_name"] == 'compute_transitive_attention':
                fn = compute_transitive_attention
            elif analysis_pass["function_name"] == 'generate_uniform_attention':
                fn = generate_uniform_attention
            elif analysis_pass["function_name"] == 'generate_uniform_attention_from_metadata':
                fn = generate_uniform_attention_from_metadata
            elif analysis_pass["function_name"] == 'generate_copy_cat_attention':
                fn = generate_copy_cat_attention
            elif analysis_pass["function_name"] == 'extract_single_slice':
                fn = extract_single_slice
            elif analysis_pass["function_name"] == 'extract_half_and_sum':
                fn = extract_half_and_sum
            elif analysis_pass["function_name"] == 'sum_over_1st_dimension':
                fn = sum_over_1st_dimension
            elif analysis_pass["function_name"] == 'raw_weights_last_layer':
                fn = raw_weights_last_layer
            elif analysis_pass["function_name"] == 'raw_weights_first_layer':
                fn = raw_weights_first_layer
            elif analysis_pass["function_name"] == 'generate_gaussian_attention_in_neighborhood_from_metadata':
                fn = generate_gaussian_attention_in_neighborhood_from_metadata
            elif analysis_pass["function_name"] == 'make_symmetric':
                fn = make_symmetric
            else:
                raise ValueError(f"Unknown function name: {analysis_pass['function_name']}")

            kwargs = analysis_pass.get('kwargs', {})
            if 'pass_metadata' in analysis_pass:
                kwargs['machine_metadata'] = True

            if analysis_pass.get("input_folder", None):
                input_folder = os.path.join(
                    config["output_data_folder"],
                    analysis_pass["input_folder"])
            else:
                input_folder = config["output_data_folder"]
            metadata_folder = os.path.join(
                config["output_data_folder"], 'metadata')
            derive_for_experiment(
                config,
                procedure_name=pass_name,
                input_folder=input_folder,
                metadata_folder=metadata_folder,
                derivation_function=fn,
                kwargs=kwargs)
    else:
        click.echo("No config file found. Insert one.")

@cli.command()
@click.pass_context
def getanswers(ctx):
    """Derive the answers in a dataset."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)

    if config:
        click.echo("Using config file information.")
        output_folder = config.get("output_data_folder")
        generated_text_folder = os.path.join(
            output_folder, "generated_sequence")
        answers_folder = os.path.join(
            output_folder, "answers")
        Path(answers_folder).mkdir(parents=True, exist_ok=True)
        all_generated_text = read_data_in_parallel(
            base_folder=generated_text_folder,
            file_type_extension=".txt",
            read_function=load_txt_file)
        for filename, text in all_generated_text.items():
            answer_filepath = os.path.join(answers_folder, filename + ".txt")
            # remove all the part of the prompt which is not an answer
            text = text[text.index("Question"):]
            # add the filename at the beginning of the text
            text = filename + "\n" + text
            with open(answer_filepath, 'w') as f:
                f.write(text)

    else:
        click.echo("No config file found. Insert one.")


@cli.command()
@click.pass_context
def getanswersdataset(ctx):
    """Derive the dataset with answers and provenance information.

    Provenance includes:
    - filename of the generated text
    - model name
    - number of try
    - answer
    """
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)

    if config:
        click.echo("Using config file information.")
        metadata_folder = os.path.join(
            config["output_data_folder"], "metadata")
        answers_folder = os.path.join(
            config["output_data_folder"], "answers")

        # all_machine_metadata = read_data_in_parallel(
        #     base_folder=metadata_folder,
        #     file_type_extension=".json",
        #     read_function=load_json_file)
        all_answers = read_data_in_parallel(
            base_folder=answers_folder,
            file_type_extension=".txt",
            read_function=load_txt_file)

        records = []
        for filename, answer_content in all_answers.items():
            n_try = filename.split("_")[2]
            records.append({
                "filename": filename,
                "model_name": config.get("model_name", ["unknown"])[0],
                "n_try": n_try,
                "answer": answer_content
            })
        # store the answers in a csv
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(
            config["output_data_folder"], "machine_answers.csv"), index=False)


if __name__ == '__main__':
    cli()
