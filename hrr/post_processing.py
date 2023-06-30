"""This file post-process the attention collected via HRR."""

import os
from os.path import join
import json
import click
import yaml
from tqdm import tqdm
from pymongo import MongoClient
from attwizard.script.utils_mongodb import load_all_aggregated_records
from attwizard.script.utils import load_json_file
from attwizard.script.utils import write_json_file
from pathlib import Path
import numpy as np

from typing import Dict, List, Tuple, Union, Any

from codeattention.source_code import SourceCode


# UTILITY FUNCTIONS


def calc_hover_visible_durations(events, tokens, start_time):
    """Calculate the total time a token was visible due to mouse hovering."""
    events = sorted(events, key=lambda x: x["time"])

    times_per_token = [0] * len(tokens)
    previous_tokens: set = set()
    previous_time: int = start_time
    for event in events:
        current_tokens = set(event["visibleTokens"])
        current_time = int(event["time"])
        delta = current_time - previous_time
        for idx in previous_tokens:
            times_per_token[idx] = times_per_token[idx] + delta

        previous_tokens = current_tokens
        previous_time = current_time

    return times_per_token


def get_tokens_info(task_metadata_folder, filename):
    """Get the tokenization of the given source code filename."""
    filepath = f"{task_metadata_folder}/{filename}.yaml"
    # print(f"Reading: {filepath}")
    with open(filepath, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata["tokens"]


def convert_to_codeattention_fomat(tokens):
    """Reformat tokens and feed to visualization library."""
    reformatted_tokens = []
    for i, tok in enumerate(tokens):
        reformatted_tokens.append({
            "t": tok["token"],
            "i": i,
            "l": tok["line"],
            "c": tok["column"]
        })
    return reformatted_tokens


def get_highlight_info(task_metadata_folder, filename):
    """Get which line was highlighted in the given source code."""
    with open(f"{task_metadata_folder}/{filename}.yaml", "r") as f:
        metadata = yaml.safe_load(f)
    if "buggy_line_number" in metadata:
        return [metadata["buggy_line_number"]]
    return None


def create_heatmap(
        att_vector: List[float],
        tokens: List[Dict[str, Any]],
        highlighted_lines: List[int],
        out_folder: str = None, pdf_filename: str = None):
    """Create an heatmap on source code."""
    java_sc = SourceCode(tokens)
    fig, ax = java_sc.show_with_weights(
        weights=att_vector,
        show_line_numbers=True,
        lines_highlight=[
            {"line": line, "type": "background", "color": "yellow",
                "alpha": .5}
            for line in highlighted_lines
        ] if highlighted_lines else None
    )
    fig.tight_layout()
    if out_folder and pdf_filename:
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            os.path.join(out_folder, pdf_filename))



@click.group()
@click.option(
    '--config', default=None,
    help="Pass path to yaml config file with keys: " +
         "mongo_db_url, " +
         "evaluation_folder, " +
         "att_weights_folder, " +
         "task_metadata_folder.")
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
@click.option(
    '--nickname', default=None,
    help='Nickname of the user to process.')
@click.option(
    '--with_heatmaps', is_flag=True, default=False)
@click.pass_context
def getweights(ctx, nickname, with_heatmaps):
    """Derive the attention weights for each submission of the given user.

    Note that the attention is computed as the total time that a token was
    visible during the experiment.
    """
    config = ctx.obj.get('CONFIG', None)
    click.echo("Using config file information.")
    evaluation_folder = config.get('evaluation_folder', None)
    main_task_dir = os.path.join(evaluation_folder, 'main_task')
    task_metadata_folder = config.get('task_metadata_folder', None)
    derived_data_folder = config.get('derived_data_folder', None)

    # create the folder for attention weights
    att_weight_dir = os.path.join(derived_data_folder, "att_weights")
    Path(att_weight_dir).mkdir(parents=True, exist_ok=True)

    raw_tokenizations = {}
    if with_heatmaps:
        code_heatmaps_dir = os.path.join(derived_data_folder, "code_heatmaps")
        Path(code_heatmaps_dir).mkdir(parents=True, exist_ok=True)
        info_tokenization = {}
        info_highlighted_lines = {}
    raw_records_filenames = os.listdir(main_task_dir)
    raw_records = []
    for record_filename in raw_records_filenames:
        if nickname and not record_filename.startswith(nickname):
            # skip irrelevant records
            continue
        with open(join(main_task_dir, record_filename), "r") as f:
            record = json.load(f)
            raw_records.append(record)
        c_nickname = record.get('nickname')
        filename = record["sourceFile"]
        print(f"Processing {c_nickname}: {filename}")
        unblur_events = record["hoverUnblurEvents"]
        if filename not in raw_tokenizations:
            tokens = get_tokens_info(task_metadata_folder, filename)
            raw_tokenizations[filename] = tokens
            if with_heatmaps:
                print("Preparing tokenization for codeattention...")
                codeatt_tokens = convert_to_codeattention_fomat(tokens)
                info_tokenization[filename] = codeatt_tokens
        else:
            tokens = raw_tokenizations[filename]
            if with_heatmaps:
                codeatt_tokens = info_tokenization[filename]
        attention = calc_hover_visible_durations(
            unblur_events, tokens, record["openedPageTime"])
        # convert to numpy and save as npy
        attention = np.array(attention)
        # save as numpy
        np.save(os.path.join(
                    att_weight_dir, record_filename.replace(".json", ".npy")),
                attention)

        if with_heatmaps:
            if filename not in info_highlighted_lines:
                highlighted_lines = get_highlight_info(
                    task_metadata_folder, filename)
                info_highlighted_lines[filename] = highlighted_lines
            else:
                highlighted_lines = info_highlighted_lines[filename]
            print(f"Generating heatmaps for {c_nickname}: {filename}")
            create_heatmap(
                att_vector=attention,
                tokens=codeatt_tokens,
                highlighted_lines=highlighted_lines,
                out_folder=code_heatmaps_dir,
                pdf_filename=record_filename.replace(".json", ".pdf"))

    print("Finished human attention processing.")


if __name__ == '__main__':
    cli()
