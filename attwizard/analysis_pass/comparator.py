"""This file compare Human vs Machine attention."""

import os
from os.path import join

from typing import Callable, List, Dict, Any
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
import torch
import click
import yaml
from transformers import AutoTokenizer

from attwizard.script.utils import get_model_folder_path, read_data_in_parallel
from attwizard.script.utils import load_json_file

from attwizard.analysis_pass.data_helper import explode_column_with_list_of_tuples
from attwizard.analysis_pass.transformation_functions import get_transf_fn
from attwizard.analysis_pass.comparison_functions import get_comp_fn


@click.group()
@click.option(
    '--config', default=None,
    help="Pass path to yaml config file with keys: " +
         "human_att_folder, " +
         "machine_att_folder, " +
         "comparisons.")
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


def get_source_file_from_filename_machine(filename: str) -> str:
    """Extract the task filename from the name of the machine file."""
    # get basename of the file
    res = filename.split(os.sep)[-1]
    return "_".join(res.split('_')[:2])


def get_name_from_human_filename(filename: str) -> str:
    """Extract the name of the user and the task from the filename."""
    res = filename.split(os.sep)[-1]
    res = "_".join(res.split('_')[:2])
    res = res.replace(".json", "")
    return res


def get_comparison_type_based_on_dims(human_data, machine_data):
    """Get the comparison type based on the dims (i.e. vector or matrix)."""
    if human_data.ndim <= 1 and machine_data.ndim <= 1:
        return "vector"
    elif human_data.ndim == 2 and machine_data.ndim == 2:
        return "matrix"
    else:
        raise ValueError(
            f"Human data has {human_data.ndim} dims and machine data " +
            f"has {machine_data.ndim} dims.")


def compare_data(
        human_data: Dict[str, any],
        human_metadata: Dict[str, any],
        machine_data: Dict[str, any],
        machine_metadata: Dict[str, any],
        comparison: Dict[str, str]) -> None:
    """Compare the Human vs Machine vectors."""
    print(f"Comparing data... ")
    print(f"n of human data: {len(human_data)}")
    print(f"n of machine data: {len(machine_data)}")
    all_comparison_records = []
    for unique_data_id in tqdm(human_data.keys()):
        c_human_metadata = human_metadata[unique_data_id]
        c_file_name = os.path.basename(c_human_metadata["source_code"])
        # handle the case where we have more than one machine attention
        # for the same filename
        relevant_machine_data_names = [
            name for name in machine_data.keys()
            if name.startswith(c_file_name)]
        if len(relevant_machine_data_names) == 0:
            click.echo(f"No machine data for {c_file_name}")
        for machine_data_name in relevant_machine_data_names:
            c_human_data = human_data[unique_data_id].copy()
            c_machine_data = machine_data[machine_data_name].copy()
            #print(f"Comparing {c_file_name}")
            c_machine_metadata = machine_metadata[machine_data_name]
            # transform the data before the comparison
            transformations = comparison["transformations"]
            for transformation in transformations:
                #print(f"Transforming data with {transformation}")
                #print("Input shape: ", c_human_data.shape)
                transformation_fn = get_transf_fn(transformation["name"])
                kwargs = transformation.get("kwargs", {})
                c_human_data, c_machine_data = transformation_fn(
                    c_human_data, c_machine_data,
                    c_human_metadata, c_machine_metadata, **kwargs)
                # print("transformation: ", transformation["name"])
                # print("Output Human shape: ", c_human_data.shape)
                # print("Output Machine shape: ", c_machine_data.shape)

            comparison_function_name = comparison["comparison_function"]
            # compare the data
            comparison_fn = get_comp_fn(comparison_function_name)
            # print(f"Content machine (before tranformation): {c_machine_data[:5,:5]}")
            # check if there are kwargs

            kwargs = comparison.get("comparison_kwargs", {})
            # create a copy
            kwargs = kwargs.copy()
            # check if the comparison function takes the metadata
            if "pass_metadata" in kwargs.keys():
                kwargs["human_metadata"] = c_human_metadata
                kwargs["machine_metadata"] = c_machine_metadata
                # remove pass_metadata from kwargs
                kwargs.pop("pass_metadata")
            comparison_res = comparison_fn(
                c_human_data, c_machine_data, **kwargs)
            # comparison type
            cmp_type = get_comparison_type_based_on_dims(
                c_human_data, c_machine_data)
            time_ts = time.time()
            time_human = datetime.fromtimestamp(time_ts).strftime(
                '%Y-%m-%d %H:%M:%S')
            transformation_names = {
                f"transformation_{i}": {
                    "name": transformation["name"],
                    "kwargs": transformation.get("kwargs", {})
                }
                for i, transformation in enumerate(transformations)}
            comparison_record = {
                "comparison_name": comparison["name"],
                "comparison_function": comparison_function_name,
                "source_code": c_file_name,
                "user_name": c_human_metadata["user_name"],
                "task_number": c_human_metadata["task_number"],
                "input_human_filestem": unique_data_id,
                "input_machine_filestem": machine_data_name,
                **transformation_names,
                "comparison_type": cmp_type,
                **comparison_res,
                "provenance": comparison,
                "time_ts_sec": time_ts,
                "time_human": time_human
            }
            all_comparison_records.append(comparison_record)
    print("Total number of comparisons: ", len(all_comparison_records))
    # save the comparison results
    df = pd.json_normalize(all_comparison_records)
    df = explode_result_column(df, comparison=comparison)
    return df


def get_all_matching_pairs(
        human_data_a: Dict[str, any],
        human_metadata_a: Dict[str, any],
        human_data_b: Dict[str, any],
        human_metadata_b: Dict[str, any],
        matching_field_a: str = 'source_code',
        matching_field_b: str = 'source_code',
        matching_function: Callable = lambda x, y: x == y,):
    """Get all the matching data based on the matching function and fields.

    Return all those pairs of unique_data_id for which the mtching function
    returns true on the two matching fields."""
    all_pairs = []
    for unique_data_id_a in human_data_a.keys():
        c_human_metadata_a = human_metadata_a[unique_data_id_a]
        field_a = c_human_metadata_a[matching_field_a]
        for unique_data_id_b in human_data_b.keys():
            c_human_metadata_b = human_metadata_b[unique_data_id_b]
            field_b = c_human_metadata_b[matching_field_b]
            if matching_function(field_a, field_b):
                all_pairs.append((unique_data_id_a, unique_data_id_b))
    return all_pairs


def compare_human_data(
        human_data_a: Dict[str, any],
        human_metadata_a: Dict[str, any],
        human_data_b: Dict[str, any],
        human_metadata_b: Dict[str, any],
        comparison: Dict[str, str],
        matching_field_a: str = 'source_code',
        matching_field_b: str = 'source_code',
        matching_function: Callable = lambda x, y: x == y,) -> None:
    """Compare Human data vs Human data on the same questions."""
    print(f"Comparing human data... ")
    print(f"n of human data a: {len(human_data_a)}")
    print(f"n of human data b: {len(human_data_b)}")
    all_pairs = get_all_matching_pairs(
        human_data_a, human_metadata_a,
        human_data_b, human_metadata_b,
        matching_field_a, matching_field_b, matching_function)
    # remove matches whith themselves
    all_pairs = [pair for pair in all_pairs if pair[0] != pair[1]]
    print(f"n of matching pairs: {len(all_pairs)}")
    all_comparison_records = []
    transformations = comparison["human"].get("transformations", None)
    if transformations is None:
        # if the human do not have any custom transformations, then
        # we use the transformation of the machine
        print("Warning: Using machine transformations also on machine data")
        print('if this is not what you want set the field "transformations"')
        print('in the "human" field of the comparison to []')
        transformations = comparison["transformations"]
    for unique_data_id_a, unique_data_id_b in tqdm(all_pairs):
        c_human_metadata_a = human_metadata_a[unique_data_id_a]
        c_human_metadata_b = human_metadata_b[unique_data_id_b]
        c_file_name = c_human_metadata_a["source_code"]
        c_human_data_a = human_data_a[unique_data_id_a].copy()
        c_human_data_b = human_data_b[unique_data_id_b].copy()
        # create two fields on the metadata b to make it appear as if it was
        # a machine data. Add the `text_prompt` and `tokens_prompt` fields.
        c_human_metadata_b["text_prompt"] = c_human_metadata_b["raw_text"]
        tokenizer_name = comparison['human']["tokenizer"]
        model_folder = comparison['human']["model_folder"]
        model_folder_path = get_model_folder_path(
            model_folder=model_folder, hugging_short_repo=tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        tmp = tokenizer(
            c_human_metadata_b["text_prompt"], return_tensors="pt")
        tokens = tmp['input_ids'][0].tolist()
        # remove the initial special token of incoder
        if "incoder" in tokenizer_name:
            tokens = tokens[1:]
        tokens_model = [tokenizer.decode([t]) for t in tokens]
        c_human_metadata_b["tokens_prompt"] = tokens_model
        c_human_metadata_b["mode_name"] = tokenizer_name
        c_human_metadata_b["config_options"] = {
            'local_model_folder': model_folder}
        for transformation in transformations:
            #print(f"Transforming data with {transformation}")
            #print("Input shape: ", c_human_data.shape)
            if transformation["name"] == "drop_first_model_token":
                continue
            transformation_fn = get_transf_fn(transformation["name"])
            kwargs = transformation.get("kwargs", {})
            c_human_data_a, c_human_data_b = transformation_fn(
                c_human_data_a, c_human_data_b,
                c_human_metadata_a, c_human_metadata_b, **kwargs)

        comparison_function_name = comparison["comparison_function"]
        # compare the data
        comparison_fn = get_comp_fn(comparison_function_name)
        # print(f"Content machine (before tranformation): {c_machine_data[:5,:5]}")
        # check if there are kwargs

        kwargs = comparison.get("comparison_kwargs", {})
        # create a copy
        kwargs = kwargs.copy()
        # check if the comparison function takes the metadata
        if "pass_metadata" in kwargs.keys():
            kwargs["human_metadata"] = c_human_metadata_a
            kwargs["machine_metadata"] = c_human_metadata_b
            # remove pass_metadata from kwargs
            kwargs.pop("pass_metadata")
        comparison_res = comparison_fn(
            c_human_data_a, c_human_data_b, **kwargs)
        # compare the data
        comparison_fn = get_comp_fn(comparison_function_name)
        # print(f"Content machine (before tranformation): {c_machine_data[:5,:5]}")
        comparison_res = comparison_fn(c_human_data_a, c_human_data_b, **kwargs)
        # comparison type
        cmp_type = get_comparison_type_based_on_dims(
            c_human_data_a, c_human_data_b)
        time_ts = time.time()
        time_human = datetime.fromtimestamp(time_ts).strftime(
            '%Y-%m-%d %H:%M:%S')
        transformation_names = {
            f"transformation_{i}": {
                "name": transformation["name"],
                "kwargs": transformation.get("kwargs", {})
            }
            for i, transformation in enumerate(transformations)}
        comparison_record = {
            "comparison_name": comparison["name"],
            "comparison_function": comparison_function_name,
            "source_code": c_file_name,
            "user_name_a": c_human_metadata_a["user_name"],
            "task_number_a": c_human_metadata_a["task_number"],
            "user_name_b": c_human_metadata_b["user_name"],
            "task_number_b": c_human_metadata_b["task_number"],
            "human_a": unique_data_id_a,
            "human_b": unique_data_id_b,
            **transformation_names,
            "comparison_type": cmp_type,
            **comparison_res,
            "provenance": comparison,
            "time_ts_sec": time_ts,
            "time_human": time_human
        }
        all_comparison_records.append(comparison_record)
    print(f"n of comparison records: {len(all_comparison_records)}")
    # save the comparison results
    df = pd.json_normalize(all_comparison_records)
    df = explode_result_column(df, comparison=comparison)
    return df


def explode_result_column(
        df: pd.DataFrame, comparison: Dict[str, Any]) -> pd.DataFrame:
    """Explode a column of the dataframe."""
    if 'comparison_output_subcolumns' not in comparison.keys():
        return df
    col_to_explode = comparison["comparison_output_column"]
    new_column_names = comparison["comparison_output_subcolumns"]
    df = explode_column_with_list_of_tuples(
        df, [col_to_explode], new_column_names=new_column_names)
    return df


@cli.command()
@click.pass_context
def compare(ctx):
    """Compare the attention weights."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)
    if config:
        click.echo("Using config file information.")
        output_comparison_folder = config["output_comparison_folder"]
        machine_att_folder = config['machine_att_folder']
        human_att_folder = config['human_att_folder']
        # read metadata
        click.echo("Loading metadata...")
        machine_metadata = read_data_in_parallel(
            base_folder=os.path.join(machine_att_folder, "metadata"),
            file_type_extension=".json",
            read_function=load_json_file)
        human_metadata = read_data_in_parallel(
            base_folder=os.path.join(human_att_folder, "metadata"),
            file_type_extension=".json",
            read_function=load_json_file)

        comparisons = config['comparisons']
        # load all machine files in a dict
        click.echo("Pre-loading human and machine attention files...")
        all_machine_data = {}
        all_human_data = {}
        for comparison in comparisons:
            cmp_human = comparison['human']
            if not cmp_human["subfolder"] in all_human_data.keys():
                all_human_data[cmp_human["subfolder"]] = \
                    read_data_in_parallel(
                        base_folder=os.path.join(
                            human_att_folder, cmp_human["subfolder"]),
                        file_type_extension=cmp_human["file_extension"])
                click.echo(f'{cmp_human["subfolder"]} (human) loaded.')
            cmp_machine = comparison['machine']
            if not isinstance(cmp_machine, list):
                cmp_machine = [cmp_machine]
            for machine_section in cmp_machine:
                if machine_section["subfolder"] not in all_machine_data.keys():
                    all_machine_data[machine_section["subfolder"]] = \
                        read_data_in_parallel(
                            base_folder=os.path.join(
                                machine_att_folder,
                                machine_section["subfolder"]),
                            file_type_extension=machine_section["file_extension"])
                    first_key = list(all_machine_data[machine_section["subfolder"]].keys())[0]
                    print(f'Shape of one element: {all_machine_data[machine_section["subfolder"]][first_key].shape}')
                    click.echo(
                        f'{machine_section["subfolder"]} (machine) loaded.')

        # create comparison folder if it does not exist
        Path(output_comparison_folder).mkdir(parents=True, exist_ok=True)

        for comparison in comparisons:
            cmp_human = comparison['human']

            machine_sections = comparison['machine']
            # check if a list, if not create a singleton list
            if not isinstance(machine_sections, list):
                machine_sections = [machine_sections]

            for cmp_machine in machine_sections:
                print("=" * 80)
                print(f"{comparison['name']}{cmp_machine['suffix']}")
                df_comparison = compare_data(
                    human_data=all_human_data[cmp_human["subfolder"]],
                    human_metadata=human_metadata,
                    machine_data=all_machine_data[cmp_machine["subfolder"]],
                    machine_metadata=machine_metadata,
                    comparison=comparison)
                # add column with the human and machine folders
                df_comparison["human_att_folder"] = human_att_folder
                df_comparison["machine_att_folder"] = machine_att_folder
                df_comparison['comparison_name'] = \
                    f"{comparison['name']}{cmp_machine['suffix']}"
                df_comparison.to_csv(
                    os.path.join(
                        output_comparison_folder,
                        f"{comparison['name']}{cmp_machine['suffix']}.csv"),
                    index=False)
                print("=" * 80)
            # compare human data with each other, using permutations
            # so that the humans are considered in both cases
            print("=" * 80)
            print(f"Humans vs Humans of: {comparison['name']}")
            df_human_comparison = compare_human_data(
                human_data_a=all_human_data[cmp_human["subfolder"]],
                human_metadata_a=human_metadata,
                human_data_b=all_human_data[cmp_human["subfolder"]],
                human_metadata_b=human_metadata,
                comparison=comparison)
            df_human_comparison["human_att_folder"] = human_att_folder
            df_human_comparison['comparison_name'] = \
                f"{comparison['name']}{cmp_human['suffix']}"
            df_human_comparison.to_csv(
                os.path.join(
                    output_comparison_folder,
                    f"{comparison['name']}{cmp_human['suffix']}.csv"),
                index=False)
    else:
        click.echo("No config file found. Insert one.")


if __name__ == '__main__':
    cli()
