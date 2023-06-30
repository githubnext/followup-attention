"""This file contains post processing functions for eye tracking data."""
import os
from typing import List, Tuple, Any, Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import click
import yaml
import json
import re
import numpy as np
from pprint import pprint
import torch
from tqdm import tqdm
import pathlib
import hashlib

from eye_tracking.utils import read_textual_snapshot
from eye_tracking.utils import read_eye_tracker_fixations
from eye_tracking.utils import offset_time_based_on_time_column
from eye_tracking.utils import get_video_path
from eye_tracking.utils import get_tot_n_frames
from eye_tracking.utils import get_video_image
from eye_tracking.utils import get_code_coordinates
from eye_tracking.utils import get_char_coordinates
from eye_tracking.utils import convert_char_coordinate_to_attention_area
from eye_tracking.utils import convert_char_attribution_to_tokens_and_weights

from attwizard.script.utils import read_data_in_parallel
from attwizard.script.utils_model import get_model_tokenization
from attwizard.normalizer import norm_convert_row_to_prob
from attwizard.visualizer.matrix import visualize_followup_graph_side_by_side
from attwizard.shaper import aggregate_dim_tokens_to_line

from attwizard.aligner import get_tokens_with_col_and_line
from attwizard.aligner import convert_weigths_from_tok_to_tok
from transformers import AutoTokenizer
from attwizard.script.utils import get_model_folder_path

from codeattention.source_code import SourceCode


@click.group()
@click.argument("config_file")
@click.option("--debug", is_flag=True, default=False)
@click.pass_context
def cli(ctx, config_file, debug):
    """Process the eye tracking data."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    click.echo(f"Reading config file:")
    pprint(config)
    ctx.ensure_object(dict)
    ctx.obj['CONFIG'] = config
    ctx.obj['DEBUG'] = debug


def get_batch_specific_folders(
        config,
        usename: str,
        participants: List[Dict[str, Any]],
        dir_participants: str,
        dir_questions: str):
    """Return batch specific folder if present, otherwise return global ones."""
    # in case of data specified in batches
    c_dir_participants = dir_participants
    c_dir_questions = dir_questions
    if "batches" in config.keys():
        # get the folder of the samples and the participants data
        try:
            c_participant_batch = int([
                p for p in participants
                if p["name"] == usename][0]["batch_no"])
            print(c_participant_batch)
            c_batch_info = [
                b for b in config['batches']
                if int(b["batch_no"]) == int(c_participant_batch)][0]
            c_dir_participants = c_batch_info["folder_participants"]
            c_dir_questions = c_batch_info["folder_questions"]
        except Exception as e:
            print(e)
            raise e
            print(
                "No batch-specific folders found."
                "using the `folder_participants` and " +
                "`folder_questions` specified on the top-level." +
                "when batch have not specified folders.")
    return c_dir_participants, c_dir_questions


###############################################################################
# ATTENTION MAPS from FIXATION DATA
###############################################################################


def visualize_task_data_over_time(df_txt_and_eye: pd.DataFrame):
    """Visualize the eye tracking and plugin data over time."""
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.histplot(
        data=df_txt_and_eye,
        x="time",
        hue="source",
        ax=ax
    )


def check_if_task_exists(
        username: str, task_number: int, output_folder: str):
    """Check if the output for this user task was already saved."""
    all_files = os.listdir(output_folder)
    for file in all_files:
        if file.startswith(f"{username}_{task_number}"):
            return True
    return False


def read_and_synch_eye_and_text_data(
        user_folder: str,
        task_number: int,
        config_manual_tabs: List[Dict[str, Any]]):
    """Read and sync eye and text data coming form the VScode plugin."""
    # vscode plugin data
    df_text = read_textual_snapshot(
        user_folder=user_folder,
        config_manual_tabs=config_manual_tabs)
    if len(df_text) == 0:
        raise FileNotFoundError(
            "No textual data found (malfunction of the VScode plugin)")
    # eye tracker data
    df_eye = read_eye_tracker_fixations(
        user_folder=user_folder, task_nr=task_number)
    df_eye = offset_time_based_on_time_column(df_eye)
    # Note that FPOGS_abs is in milliseconds
    # but the FPOFD is in seconds
    #df_eye = df_eye.rename(columns={"time_abs": "time"})
    df_eye = df_eye.rename(columns={"FPOGS_abs": "time"})
    # merge data
    eye_events = df_eye[
        ["time", "FPOGX", "FPOGY", "FPOGD"]].to_dict(orient="records")
    eye_events = [{"source": "eye", **e} for e in eye_events]
    text_events = df_text[
        ["time", "source_filename", "content_lines"]].to_dict(orient="records")
    text_events = [{"source": "text", **e} for e in text_events]
    all_events = eye_events + text_events
    df_txt_and_eye = pd.DataFrame.from_records(all_events)
    # keep only the text data in between eye tracking data
    # because the rest refer to a different task
    first_ts_eye = df_eye.sort_values("time").iloc[0]["time"]
    last_ts_eye = df_eye.sort_values("time").iloc[-1]["time"]
    # filter and keep the text data with time between the first and last ts eye
    df_txt_and_eye = df_txt_and_eye[
        (df_txt_and_eye["time"] >= first_ts_eye) &
        (df_txt_and_eye["time"] <= last_ts_eye)
    ]
    return df_txt_and_eye


def screen_coordinates_from_video(
        user_name: str,
        task_number: int,
        user_folder: str,
        markers_info: Dict[str, Any],
        custom_frame: int = None):
    """Get the screen coordinates from the video."""
    video_path = None
    video_nr = task_number
    # in some case we might have only the fixation data and no video
    # for those case we assume that the screen location is the same
    # of the previous task, thus we inspec that video.
    # e.g. no video for file task 2, then we check the video of task 1
    # and so on.
    while video_path is None and video_nr >= 0:
        try:
            video_path = get_video_path(
                user_folder=user_folder, task_nr=video_nr)
        except IndexError as e:
            video_nr -= 1
            if video_nr < 0:
                click.echo(f"No video found for user: {user_name}")
                click.echo("Impossible to infer the screen coordinate")
                return None
    suffix = ""
    if video_nr < task_number:
        click.echo(f"No video for task {task_number}.")
        click.echo(
            f"Using the coordinate of the video of task {video_nr} instead")
        suffix = "_video_" + str(video_nr)

    data = get_tot_n_frames(video_path)
    tot_n_frames = data["tot_n_frames"]
    if custom_frame is None:
        # get the middle frame
        custom_frame = int(tot_n_frames // 2)
    img, img_path = get_video_image(
        video_path=video_path,
        frame_number=custom_frame, output_folder="./tmp")
    # screen coordinates of the code area
    code_screen_coordinate_abs = get_code_coordinates(
        screenshot_path=img_path,
        percentage_format=False,
        debug=False,
        **markers_info)
    click.echo(f"code_screen_coordinate_abs: {code_screen_coordinate_abs}")
    return code_screen_coordinate_abs, suffix


def get_most_popular_file_in_IDE(df_textual_data: pd.DataFrame):
    """Returns the filename that occures most often in text snapshots."""
    df = df_textual_data.copy()
    df["file_name"] = df["source_filename"].apply(
        lambda filepath: re.search(
            "([a-zA-Z0-9 _.]+):[0-9]+\n$", filepath).group(1))
    df["folder"] = df["source_filename"].apply(
        lambda filepath: re.search(
            "\\\\([a-zA-Z]+)\\\\[a-zA-Z0-9 _.]+:[0-9]+\n$",
            filepath).group(1))
    # join the folder and filename
    df["file_path"] = df.apply(
        lambda row: os.path.join(row["folder"], row["file_name"]), axis=1)
    # get the most frequent file_path value
    most_frequent_file_path = df["file_path"].value_counts().index[0]
    return most_frequent_file_path


def get_human_attention_from_eye_tracking(
        unique_data_id: str,
        user_name: str,
        task_number: int,
        dir_participants: str,
        dir_questions: str,
        config_zoom: Dict[str, Any],
        config_attention: Dict[str, Any],
        config_screen_res: Dict[str, Any],
        config_manual_tabs: List[Dict[str, Any]],
        output_folder: str = None,
        manual_screen_coordinates: Dict[str, int] = None,
        override_everything: bool = False,
        debug: bool = False):
    """Derive the human attention from eye tracking data.

    Note that this method does the following assumptions:
    - the code area is never resized during the task and the zoom is never
        changed (the grid of characters is sampled in the mid frame of the
        video recording using two image markers)
    - the main code displayed in current task is that shown in the middle of
        the task recording.
    - we assume that for the entire task duration, the code area is always
        visible.
    - we consider the start of the task when the eye tracking data start to
        be recorded for the given task. And the end when we do not see any
        eye tracking data anymore.
    - we consider only fixation points, no gaze points. Thus we can have some
        ara of the code which are never attended by fixation, but perhaps the
        person only quickly scanned through it.

    If an output folder is passed, this method will produce the following
    files:
    1. a .json file containing the metadata of the given task
    2. a .png file containing the heatmap of the attention area with all the
        attention weight, also on characters which are not in the original
        input such as the blank spaces at the end of the line or the spaces in
        between two lines.
    3. a .png file containing the heatmap of the attention area only on the
        characters in the original input. Important: this will not consider
        the attention contributed by eye fixations on the blank lines or spaces
        at all.

    Note that all the output files follow the naming convention:
    <username>_<task_number>...

    """
    if output_folder and not override_everything:
        if check_if_task_exists(user_name, task_number, output_folder):
            click.echo("Result already existing. Skip.")
            return

    user_folder = os.path.join(dir_participants, user_name)
    # READ AND SYNCH DATA: EYE AND TEXTUAL DATA
    try:
        df_txt_and_eye = read_and_synch_eye_and_text_data(
            user_folder=user_folder,
            task_number=task_number,
            config_manual_tabs=config_manual_tabs)
    except FileNotFoundError as e:
        print(e)
        return
    # visualize available data
    if debug:
        visualize_task_data_over_time(df_txt_and_eye)
        plt.show()

    markers_info = config_zoom.copy()
    markers_info.pop("n_lines")
    markers_info.pop("n_col")

    if manual_screen_coordinates:
        code_screen_coordinate_abs = manual_screen_coordinates
        suffix = "_manual"
    else:
        code_screen_coordinate_abs, suffix = screen_coordinates_from_video(
            user_name=user_name,
            task_number=task_number,
            user_folder=user_folder,
            markers_info=markers_info)
        # print(f"user_name: {user_name}")
        # print(f"task_number: {task_number}")
        # print("code_screen_coordinate_abs")
        # print(code_screen_coordinate_abs)

    screen_info = {}
    screen_info["n_lines"] = config_zoom["n_lines"]
    screen_info["n_col"] = config_zoom["n_col"]

    df_textual_data = df_txt_and_eye[df_txt_and_eye["source"] == "text"]

    target_file_path = get_most_popular_file_in_IDE(
        df_textual_data=df_textual_data)
    click.echo(f'File of the task: {target_file_path}')
    n_lines_original_file = len(
        open(os.path.join(dir_questions, target_file_path)).readlines())

    current_text_snapshot = None
    char_attribution_list = []
    # scan the data sequentially in order of time
    for i, event in df_txt_and_eye.sort_values(by="time").iterrows():

        # every time we get a new text event, we update the text snapshot
        if event["source"] == "text":
            current_text_snapshot = event
        # every time we get a new eye event, we compute the attention span
        # since the last eye event
        elif event["source"] == "eye":
            if current_text_snapshot is None:
                continue
            time_elapsed_on_this_fixation = event["FPOGD"]

            fixation_X = event["FPOGX"]
            fixation_Y = event["FPOGY"]
            snapshot_line_start = int(
                re.search(":([0-9]+)\n$",
                          current_text_snapshot["source_filename"]).group(1))
            #print(f"snapshot_line_start: {snapshot_line_start}")

            (main_char_line_index, main_char_col_index) = get_char_coordinates(
                eye_x_perc=fixation_X, eye_y_perc=fixation_Y,
                debug=False,
                **config_screen_res,
                **code_screen_coordinate_abs,
                **screen_info)
            assert main_char_line_index >= 0, \
                "line index is negative, it must be greater or equal than 0"
            assert main_char_col_index >= 0, \
                "col index is negative, it must be greater or equal than 0"
            assert main_char_line_index < len(open(os.path.join(dir_questions, target_file_path)).readlines()), \
                "line must be less or equal than the number of lines in the file"
            char_coordinates = convert_char_coordinate_to_attention_area(
                line=main_char_line_index + 1,
                col=main_char_col_index + 1,
                content="".join(
                    current_text_snapshot["content_lines"]),
                content_starting_line=snapshot_line_start,
                tot_screen_col=config_zoom["n_col"],
                tot_screen_lines=config_zoom["n_lines"],
                **config_attention,
                # debug=debug
            )
            # add time info to all the coordinates
            char_coordinates = [
                {
                    "line": int(c[0]),
                    "column": int(c[1]),
                    "time": event["time"],
                    "time_spent": time_elapsed_on_this_fixation}
                for c in char_coordinates]
            char_attribution_list.extend(char_coordinates)

    df_char_attribution = pd.DataFrame.from_records(char_attribution_list)
    # include positions never attended
    # this coefficients are generous to enlarge the admitted attention area
    # to ensure that all the glances of the developers are included
    # even if they are very far to the right or very far to the bottom
    # after the end of the file
    max_width = int(df_char_attribution["column"].max() * 2)
    max_height = int(df_char_attribution["line"].max() + 5)
    matrix = np.zeros((max_height, max_width))

    for i, row in df_char_attribution.iterrows():
        matrix[
            int(row["line"] - 1), int(row["column"] - 1)] += row["time_spent"]

    # get the position of row and column of the zero values in the matrix
    zero_positions = np.where(matrix == 0)
    positions_never_attended = [
        {"line": int(c[0]) + 1, "column": int(c[1]) + 1,
         "time_spent": 0, "time": 0}
        for c in zip(zero_positions[0], zero_positions[1])]

    full_list = char_attribution_list + positions_never_attended
    # return char_attribution_list, positions_never_attended
    df_char_attribution_with_spaces = pd.DataFrame.from_records(full_list)
    df_char_attribution_with_spaces = \
        df_char_attribution_with_spaces.sort_values(by=["line", "column"])

    task_id = f"{user_name}_{task_number}{suffix}"

    source_file_path = os.path.join(dir_questions, target_file_path)
    char_level_tokens, att_weights = \
        convert_char_attribution_to_tokens_and_weights(
            df_char_attribution=df_char_attribution_with_spaces,
            source_code_file_path=source_file_path,
            tot_screen_lines=int(config_zoom["n_lines"]),
            tot_screen_col=int(config_zoom["n_col"]),
        )
    sorted_tokens = sorted(char_level_tokens, key=lambda x: x["i"])

    my_source_code = SourceCode(sorted_tokens)
    click.echo("Creating att. heatmap on all chars, " +
               "including additional space char artificially add " +
               "in between lines.")
    fig, ax = my_source_code.show_with_weights(
        weights=att_weights, show_line_numbers=True)

    if output_folder:
        # create heatmap folder if it doesn't exist
        pathlib.Path(os.path.join(output_folder, "code_heatmap_w_spaces")).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(output_folder, "code_heatmap_w_spaces", f"{unique_data_id}.png"))

        # keep only attention weights on characters which are present also in
        # the original file
        with open(source_file_path, "r") as f:
            original_content_lines = f.readlines()
            original_content = "".join(original_content_lines)
        source_code_positions = []
        # TODO: avoid double for
        # from pprint import pprint
        # pprint(original_content_lines)
        for i, line in enumerate(original_content_lines):
            for j, char in enumerate(line):
                n_line = i + 1
                n_col = j + 1
                source_code_positions.append({"line": n_line, "column": n_col})
        df_original_chars = pd.DataFrame.from_records(source_code_positions)
        # check the two new line chars
        # print("df_char_attribution_with_spaces")
        # print(df_char_attribution_with_spaces[
        #     (df_char_attribution_with_spaces["line"] == 1) &
        #     (df_char_attribution_with_spaces["column"] == 98)])


        # print("original")
        # print(df_original_chars[
        #     (df_original_chars["line"] == 1) &
        #     (df_original_chars["column"] == 98)])
        # print(df_original_chars[df_original_chars["line"] == 2])
        # print(df_original_chars[df_original_chars["line"] == 3])
        df_char_attribution_original_char_only = pd.merge(
            right=df_char_attribution_with_spaces,
            left=df_original_chars,
            on=["line", "column"],
            how='inner')

        assert df_char_attribution_original_char_only["column"].min() >= 1, \
            "column index must be greater or equal than 1"
        assert df_char_attribution_original_char_only["line"].min() >= 1, \
            "line index must be greater or equal than 1"
        assert df_char_attribution_original_char_only["line"].max() < n_lines_original_file+ 1, \
            "line index must be less than the number of lines in the file (+1 " + \
            "because we start counting from 1)"
        # check the two new line chars
        # print("after")
        # print(df_char_attribution_original_char_only[
        #     (df_char_attribution_original_char_only["line"] == 1) &
        #     (df_char_attribution_original_char_only["column"] == 98)])
        # print(df_char_attribution_original_char_only[df_char_attribution_original_char_only["line"] == 2])
        # print(df_char_attribution_original_char_only[df_char_attribution_original_char_only["line"] == 3])

        # only original file chars
        char_level_tokens, att_weights = \
            convert_char_attribution_to_tokens_and_weights(
                df_char_attribution=df_char_attribution_original_char_only,
                source_code_file_path=source_file_path,
                tot_screen_lines=int(config_zoom["n_lines"]),
                tot_screen_col=int(config_zoom["n_col"]))
        my_source_code = SourceCode(char_level_tokens)
        click.echo("Creating att. heatmap including only chars" +
                   "which are also present in the original source file")
        fig, ax = my_source_code.show_with_weights(
            weights=att_weights, show_line_numbers=True)
        pathlib.Path(os.path.join(output_folder, "code_heatmap")).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(output_folder, "code_heatmap", f"{unique_data_id}.png"))

        # save char level tokens and weights to a json file
        # create the metadata folder if it does not exist
        pathlib.Path(os.path.join(output_folder, "metadata")).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_folder, "metadata", f"{unique_data_id}.json"), "w") as f:
            metadata = {
                "user_name": user_name,
                "task_number": task_number,
                "source_code": target_file_path,
                "tokenization": char_level_tokens,
                "raw_text": original_content
            }
            json.dump(metadata, f)
        # create attention weight folder
        pathlib.Path(os.path.join(output_folder, "att_weights")).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(output_folder, "att_weights", f"{unique_data_id}.npy"), att_weights)
        assert len(char_level_tokens) == len(att_weights), \
            f"len(char_level_tokens) != len(att_weights) - " + \
            f"{len(char_level_tokens)} != {len(att_weights)} " + \
            "the process to convert token-level attention to char-level " + \
            "attention failed."
        assert len(char_level_tokens) == len(original_content), \
            f"len(char_level_tokens) != len(original_content) - " + \
            f"{len(char_level_tokens)} != {len(original_content)} " + \
            "the process to convert token-level attention to char-level " + \
            "attention failed."
    else:
        plt.show()

    if output_folder:
        click.echo("Save the user interactions as char-level events")
        # create intermediat folder
        pathlib.Path(os.path.join(output_folder, "intermediate_char_events")).mkdir(parents=True, exist_ok=True)
        df_char_attribution_original_char_only.to_csv(
            os.path.join(
                output_folder,
                "intermediate_char_events",
                f"{unique_data_id}.csv"), index=False)

    return df_char_attribution_original_char_only, char_level_tokens, att_weights  # noqa


@cli.command()
@click.pass_context
@click.option("--username", default=None)
@click.option("--zoom", default=None)
@click.option("--task_number", default=None)
@click.option("--override", is_flag=True, default=False)
def getweights(ctx, username, zoom, task_number, override):
    """Get attention weights."""
    click.echo('Extract attention weights from eye tracker data.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_participants = config['folder_participants']
    dir_questions = config['folder_questions']
    dir_output = config['folder_output']
    participants = config['participants']

    to_extract = []

    if username:
        username = str(username)
        c_user = [
            p for p in participants if p['name'] == username][0]
        if not zoom:
            zoom = c_user['zoom']
        zoom = str(zoom)
        if not task_number:
            tasks = c_user['tasks']
            task_number = tasks[0]
        task_number = int(task_number)
        to_extract.append((username, zoom, task_number))
    else:
        click.echo("Extracting all participants data.")
        for p in participants:
            for task_number in p["tasks"]:
                to_extract.append(
                    (p['name'], p['zoom'], task_number))

    data_mapping = []
    for i, (name, zoom, task_number) in enumerate(to_extract):
        click.echo("=" * 80)
        click.echo(f"Extracting: {name} (zoom: {zoom} - task: {task_number})")
        c_dir_participants, c_dir_questions = get_batch_specific_folders(
            config=config,
            usename=name,
            participants=participants,
            dir_participants=dir_participants,
            dir_questions=dir_questions)
        config_zoom = config["zoom_dependant_parameters"][zoom]
        config_attention = config["attention_parameters"]
        config_screen_res = config["screen_resolution"]
        config_manual_tabs = config["manual_tab_replacements"]
        unique_data_id = hashlib.sha256(
            (name + str(task_number)).encode('utf-8')).hexdigest()[:6]
        data_mapping.append({
            "unique_data_id": unique_data_id,
            "user": name,
            "task_number": task_number})
        # check if the screen grid should be computed on a different video
        c_participant = [
            p for p in participants if p['name'] == name][0]
        manual_screen_coordinates = c_participant.get(
            "manual_screen_coordinates", None)

        get_human_attention_from_eye_tracking(
            unique_data_id=unique_data_id,
            user_name=name,
            task_number=task_number,
            dir_participants=c_dir_participants,
            dir_questions=c_dir_questions,
            config_zoom=config_zoom,
            config_attention=config_attention,
            config_screen_res=config_screen_res,
            config_manual_tabs=config_manual_tabs,
            output_folder=dir_output,
            manual_screen_coordinates=manual_screen_coordinates,
            override_everything=override,
            debug=debug
        )
    # save data mapping to a json file
    writing_flag = 'w' if override else 'a'
    with open(os.path.join(dir_output, "data_mapping.json"), writing_flag) as f:
        json.dump(data_mapping, f)



###############################################################################
# ATTENTION MAPS from FIXATION DATA - TOKEN LEVEL
###############################################################################

@cli.command()
@click.pass_context
def derivetokenweights(ctx):
    """Get attention weights at a token level."""
    click.echo('Derive a token level attention from the char level attention (TOKEN-LEVEL).')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_output = config['folder_output']

    dir_char_level_attention = os.path.join(dir_output, 'att_weights')
    dir_metadata = os.path.join(dir_output, 'metadata')

    all_files = os.listdir(dir_char_level_attention)
    all_npy_files = [f for f in all_files if f.endswith('.npy')]

    models_folder = config['tokenization']['models_folder']
    model_names = config['tokenization']['model_names']

    for model_name in model_names:
        # create a folder for the model
        escaped_model_name = model_name.replace("/", "_")
        output_model_folder = os.path.join(
            dir_output, f'att_weights_{escaped_model_name}')
        output_model_metadata_folder = os.path.join(
            output_model_folder, 'metadata')
        # create the folder if it does not exist
        pathlib.Path(output_model_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_model_metadata_folder).mkdir(parents=True, exist_ok=True)
        # get tokenizer
        model_folder_path = get_model_folder_path(
            model_folder=models_folder,
            hugging_short_repo=model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        for c_file in all_npy_files:
            click.echo(f"Processing {c_file}")
            c_metadata_path = os.path.join(
                dir_metadata, c_file.replace('.npy', '.json'))
            with open(c_metadata_path, 'r') as f:
                c_metadata = json.load(f)
            raw_text = c_metadata['raw_text']
            tmp = tokenizer(raw_text, return_tensors="pt")
            tokens = tmp['input_ids'][0].tolist()
            # model specific hack
            if "incoder" in model_name:
                # remove the first special token
                tokens = tokens[1:]
            tokens_model = [tokenizer.decode([t]) for t in tokens]
            # tokens_model = [tokenizer.decode([t]) for t in tokens_w_special]
            tokens_char = [t for t in raw_text]
            tokenization_char = get_tokens_with_col_and_line(
                text=raw_text, tokens=tokens_char)
            tokenization_model = get_tokens_with_col_and_line(
                text=raw_text, tokens=tokens_model)
            # get the attention weights
            char_att_weights = np.load(os.path.join(
                dir_char_level_attention, c_file))
            # get the attention weights at the token level
            token_att_weights = convert_weigths_from_tok_to_tok(
                tokenization=tokenization_char,
                weights=char_att_weights,
                target_tokenization=tokenization_model,
            )
            # augment tokenization with weights
            tokenization_model = [
                {
                    **t, 'w': w,
                    'model': model_name,
                    'id': c_file.replace('.npy', '')
                }
                for t, w in zip(tokenization_model, token_att_weights)]
            # save the token level attention weights
            path_tokenization_metadata = os.path.join(
                output_model_metadata_folder, c_file.replace('.npy', '.json'))
            with open(path_tokenization_metadata, 'w') as f:
                json.dump(tokenization_model, f)
            click.echo(f"Saved {path_tokenization_metadata}")
            # save the attention weights
            np.save(os.path.join(
                output_model_folder, c_file), token_att_weights)
            click.echo(f"Saved {os.path.join(output_model_folder, c_file)}")


###############################################################################
# FOLLOW-UP ATTENTION from USER EVENTS
###############################################################################


def check_if_task_exists_followup(
        unique_data_id: str, output_folder: str):
    """Check if the output for this user task was already saved."""
    return os.path.exists(
        os.path.join(output_folder, "data_followup_tokens_tokens_model", f"{unique_data_id}.npy"))


def get_df_events_and_metadata(
        username: str,
        task_number: int,
        folder: str,
        debug: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Get the events and metadata for a given user and task."""
    try:
        target_csv_event_file = [
            f for f in os.listdir(folder)
            if f.endswith("char_sequence.csv") and
            f.startswith(f"{username}_{task_number}")
        ][0]
        target_metadata_file = [
            f for f in os.listdir(folder)
            if f.endswith("json") and f.startswith(f"{username}_{task_number}")
        ][0]
    except IndexError:
        raise FileNotFoundError(
            f"No metadata and char_sequence.csv files found for" +
            f" user {username} and task {task_number}")
    df_events = pd.read_csv(os.path.join(folder, target_csv_event_file))
    with open(os.path.join(folder, target_metadata_file), "r") as f:
        metadata = json.load(f)
    return df_events, metadata


def get_corresponding_token_id(
        column: int, line: int, formatted_tokens: List[Dict[str, Any]]) -> int:
    """Get the token id of the token at the given line and column."""
    column = int(column)
    line = int(line)
    for i in range(len(formatted_tokens)):
        if formatted_tokens[i]["l"] == line:
            if ((column >= formatted_tokens[i]["c"]) and
                    (column < formatted_tokens[i]["c"] + len(formatted_tokens[i]["t"]))):  # noqa
                return formatted_tokens[i]["i"]
    raise ValueError(f"No token found at line {line} and column {column}.")


def get_corresponding_content(
        formatted_tokens: List[Dict[str, Any]], idx: int) -> str:
    """Get the content of the token at the given index."""
    return [x["t"] for x in formatted_tokens if x["i"] == idx][0]


def save_followup_matrix_tokens(
        followup_matrix_tokens,
        unique_data_id: str,
        model_tokenization: List[Dict[str, Any]],
        save_images: bool,
        output_folder: str):
    """Save the followup matrix to a file."""
    if save_images:
        tokens = [x["t"] for x in model_tokenization]
        size = len(tokens)
        fig, ax = plt.subplots(figsize=(size/7, size/7))
        sns.heatmap(
            followup_matrix_tokens,
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cmap="Reds"
        )
        fig.tight_layout()
        click.echo(
            f"Saving {unique_data_id}.png...")

        pathlib.Path(os.path.join(
            output_folder, "img_followup_tokens_tokens")).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            os.path.join(
                output_folder,
                "img_followup_tokens_tokens",
                f"{unique_data_id}.png"))
    # save the matrix as npy
    pathlib.Path(os.path.join(
        output_folder, "data_followup_tokens_tokens")).mkdir(parents=True, exist_ok=True)
    np.save(
        os.path.join(
            output_folder,
            "data_followup_tokens_tokens",
            f"{unique_data_id}.npy"),
        followup_matrix_tokens)


def save_followup_matrix_tokens_model(
        followup_matrix_tokens,
        unique_data_id: str,
        model_tokenization: List[Dict[str, Any]],
        save_images: bool,
        output_folder: str):
    """Save the followup matrix to a file (with the model tokenization)."""
    unique_ids = list(sorted(set([x["i"] for x in model_tokenization])))
    model_tokens = []
    for i in unique_ids:
        tokens_to_group = [x for x in model_tokenization if x["i"] == i]
        # sort tokens by line and column
        tokens_to_group.sort(key=lambda x: (x["l"], x["c"]))
        # join their content
        content = "".join([x["t"] for x in tokens_to_group])
        model_tokens.append(content)
    tokens = model_tokens
    if save_images:
        size = len(tokens)
        fig, ax = plt.subplots(figsize=(size/7, size/7))
        sns.heatmap(
            followup_matrix_tokens,
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cmap="Reds"
        )
        fig.tight_layout()
        pathlib.Path(os.path.join(
            output_folder, "img_followup_tokens_tokens_model")).mkdir(parents=True, exist_ok=True)
        click.echo(
            f"Saving {unique_data_id}.png...")
        fig.savefig(
            os.path.join(
                output_folder,
                "img_followup_tokens_tokens_model",
                f"{unique_data_id}.png"))
    # create folder and subfolder for metadata
    pathlib.Path(os.path.join(
        output_folder, "data_followup_tokens_tokens_model", "metadata")).mkdir(parents=True, exist_ok=True)
    # save the matrix as npy
    np.save(
        os.path.join(
            output_folder,
            "data_followup_tokens_tokens_model",
            f"{unique_data_id}.npy"),
        followup_matrix_tokens)
    # save json with the model tokens
    with open(
            os.path.join(
                output_folder,
                "data_followup_tokens_tokens_model",
                "metadata",
                f"{unique_data_id}.json"),
            "w") as f:
        json.dump({
            "tokenization": model_tokenization,
            "model_tokens": model_tokens
        }, f)


def save_followup_matrix_lines(
        followup_matrix_lines,
        unique_data_id: str,
        raw_text: str,
        save_images: bool,
        output_folder: str):
    """Save the followup matrix LINE format to a file."""
    if save_images:
        line_contents = raw_text.split("\n")
        size = len(line_contents)
        fig, ax = plt.subplots(figsize=(size/6, size/6))
        sns.heatmap(
            followup_matrix_lines,
            xticklabels=line_contents,
            yticklabels=line_contents,
            cmap="Reds",
            ax=ax
        )
        fig.tight_layout()
        click.echo(f"Saving {unique_data_id}.png...")
        pathlib.Path(os.path.join(output_folder, "img_followup_lines_lines")).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            os.path.join(
                output_folder,
                "img_followup_lines_lines",
                f"{unique_data_id}.png"))
    # save the matrix as npy
    pathlib.Path(os.path.join(output_folder, "data_followup_lines_lines")).mkdir(parents=True, exist_ok=True)
    np.save(
        os.path.join(
            output_folder,
            "data_followup_lines_lines",
            f"{unique_data_id}.npy"),
        followup_matrix_lines)


def save_network_lines(
        follow_matrix_lines,
        unique_data_id: str,
        raw_text: str,
        output_folder: str):
    """Save the network graph LINE to LINE."""
    line_except_last = raw_text.split("\n")
    # remove last line if empty
    if line_except_last[-1] == "":
        line_except_last = line_except_last[:-1]
    fig, ax = visualize_followup_graph_side_by_side(
        adj_mat=follow_matrix_lines,
        from_seq=line_except_last,
        to_seq=line_except_last,
        multiline_labels=False
    )
    fig.tight_layout()
    click.echo("Saving the network diagram for Line --> Line")
    pathlib.Path(os.path.join(output_folder, "img_network_line_line")).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(
        output_folder, "img_network_line_line", f"{unique_data_id}.png"))


def get_followup_matrix(
        unique_data_id: str,
        decay: float,
        normalization: str,
        tokenizer: str,
        model_folder: str,
        input_folder: str,
        output_folder: str,
        override_everything: bool = False,
        force_model_tokens: bool = False,
        levels: List[str] = ["line", "token"],
        save_images: bool = True,
        debug: bool = False):
    """Generate the follow-up matrix with the given tokenizer."""

    if output_folder and not override_everything:
        if check_if_task_exists_followup(unique_data_id, output_folder):
            click.echo("Result already existing. Skip.")
            return

    path_events = os.path.join(
        input_folder, "intermediate_char_events", f"{unique_data_id}.csv")
    # note that time is in milliseconds, whereas the time_spent is in seconds
    df_events = pd.read_csv(path_events)
    path_metadata = os.path.join(
        input_folder, "metadata", f"{unique_data_id}.json")
    metadata = json.load(open(path_metadata, "r"))

    model_tokenization = get_model_tokenization(
        raw_text=metadata["raw_text"],
        tokenizer=tokenizer,
        model_folder=model_folder)

    df_events["token_id"] = df_events.apply(
        lambda row: get_corresponding_token_id(
            column=row["column"], line=row["line"],
            formatted_tokens=model_tokenization), axis=1)
    df_events["content"] = df_events.apply(
        lambda row: get_corresponding_content(
            formatted_tokens=model_tokenization, idx=row["token_id"]), axis=1)
    # keep only those with positive time
    df_events = df_events[df_events["time"] > 0]
    # drop duplicates on the same token and at the same time
    df_events = df_events.drop_duplicates(subset=["time", "token_id"])
    df_events = df_events.sort_values(by=["time"])
    n_events = len(df_events)
    n_tokens = len(model_tokenization)
    followup_matrix = torch.zeros((n_tokens, n_tokens))
    # get the corresponding token indices
    token_indices = df_events["token_id"].values
    # get the times of the events
    times = torch.tensor(
        df_events["time"].values).expand(n_events, n_events).t()
    times = times / 1000  # convert to seconds
    print("Times:", times[:5, :5])
    # get the duration of the events
    durations = torch.tensor(
        df_events["time_spent"].values).expand(n_events, n_events).t()

    # X1 is the start of the from token
    X1 = times
    print("X1:", X1[:5, :5])
    X1_end = times + durations
    print("X1_end:", X1_end[:5, :5])
    # X2 is the start of the to token
    X2 = times.t()
    print("X2:", X2[:5, :5])
    X2_end = times.t() + durations.t()
    print("X2_end:", X2_end[:5, :5])

    # get the difference between times
    diff_times = X2 - X1_end
    # keep only the positive differences
    # diff_times = torch.clamp(diff_times, min=0)
    # get a mask on the negative differences
    mask_to_remove = (diff_times < 0)
    # count how many negative differences there are
    n_neg_diff = torch.sum(mask_to_remove)
    print(f"Total entries {n_events*n_events}")
    print(f"{n_neg_diff} negative differences")
    # IMPORTANT: we consider only events where X2 is after X1_end
    # namely token 1 comes before token2
    # token 1 is the token FROM
    # token 2 is the token TO

    # def compute_integral_attention(X2, X2_end, decay):
    #     """Analytics integran evaluated in 0, diff_time."""
    #     c = -decay
    #     return (1/(c)) * (np.exp(c * X2_end) - np.exp(c * X2 ))

    # rescale all the times to start at X1_end
    X2 = X2 - X1_end
    X2_end = X2_end - X1_end

    # X2 is the start of the to token
    print("X2:", X2[:5, :5])
    print("X2_end:", X2_end[:5, :5])


    # integral definite = F(X2) - F(X1)
    c = -decay
    F_X2_end = np.exp(c * X2_end)
    F_X2 = np.exp(c * X2)
    decayed_diff = (1/c) * (F_X2_end - F_X2)
    # zero out the negative differences based on the mask
    decayed_diff[mask_to_remove] = 0

    # integrate over the time difference
    # decayed_diff = diff_times.apply_(
    #    lambda diff_time: compute_integral_attention(diff_time, decay))

    # reorder the events so that events starting token are contiguous
    # likewise for events that end at a ceraing token
    indices = torch.tensor(token_indices)
    arg_sort = torch.argsort(indices)
    arg_sort = arg_sort.contiguous()
    sorted_indices = torch.index_select(indices, 0, arg_sort)
    sorted_indices = sorted_indices.contiguous()
    to_sort_decayed_diff = decayed_diff
    # sort the rows
    sorted_row_decayed_diff = torch.index_select(
        to_sort_decayed_diff, 0, arg_sort)
    # sort the columns
    sorted_row_and_col_decayed_diff = torch.index_select(
        sorted_row_decayed_diff, 1, arg_sort)
    decayed_diff = sorted_row_and_col_decayed_diff.contiguous()

    decayed_diff = decayed_diff.to(torch.float)
    decayed_diff = torch.nan_to_num(decayed_diff, nan=0.0)

    # compute the Follow-Up Matrix for TOKENS
    click.echo("Computing the Follow-Up Matrix for TOKENS...")
    n_token_with_no_interactions = 0
    # for each pair of tokens
    for i in tqdm(range(n_tokens)):
        # compute the index of the first row
        # get the first occurrence of the token i in the sorted indices
        occurrences_of_i = (sorted_indices == i).nonzero(as_tuple=True)[0]
        if len(occurrences_of_i) == 0:
            followup_matrix[i, :] = 0
            # print(f"No occurrence of token at position {i}")
            n_token_with_no_interactions += 1
            continue
        first_row_index = occurrences_of_i[0]
        # compute the index of the last row
        last_row_index = occurrences_of_i[-1]
        for j in range(n_tokens):
            # compute the index of the first column
            occurrences_of_j = (sorted_indices == j).nonzero(as_tuple=True)[0]
            if len(occurrences_of_j) == 0:
                followup_matrix[i, j] = 0
                continue
            first_col_index = occurrences_of_j[0]
            # compute the index of the last column
            last_col_index = occurrences_of_j[-1]
            # slice the matrix between the first and last row and columns
            slice_matrix = decayed_diff[
                first_row_index:last_row_index+1,
                first_col_index:last_col_index+1]
            # compute the sum of the matrix
            strength = slice_matrix.sum()
            # store the value in the followup matrix
            followup_matrix[i, j] = strength
    print(f"We have {n_token_with_no_interactions} " +
          "tokens with no interactions.")
    # remove not a nan values
    followup_matrix = torch.nan_to_num(followup_matrix, nan=0.0)

    if force_model_tokens:
        # aggregate new lines which are mapped to the same token index
        model_tokens_indices = np.array(
            [t["i"] for t in model_tokenization]
        )
        model_like_followup_matrix = followup_matrix
        model_like_followup_matrix = torch.nan_to_num(model_like_followup_matrix, nan=0.0)
        model_like_followup_matrix = aggregate_dim_tokens_to_line(
            att_tensor=model_like_followup_matrix,
            line_indices=model_tokens_indices,
            dim=0)
        model_like_followup_matrix = aggregate_dim_tokens_to_line(
            att_tensor=model_like_followup_matrix,
            line_indices=model_tokens_indices,
            dim=1)

    if 'line' in levels:
        # compute the Follow-Up Matrix for LINES
        click.echo("Computing the Follow-Up Matrix for LINES...")
        line_indices = np.array(
            [t["l"] - 1 for t in model_tokenization]
        )
        followup_line_matrix = followup_matrix
        followup_line_matrix = torch.nan_to_num(followup_line_matrix, nan=0.0)
        followup_line_matrix = aggregate_dim_tokens_to_line(
            att_tensor=followup_line_matrix, line_indices=line_indices,
            dim=0)
        followup_line_matrix = aggregate_dim_tokens_to_line(
            att_tensor=followup_line_matrix, line_indices=line_indices,
            dim=1)

    if output_folder:

        # if normalization == "turn_lines_into_prob_distrib":
        #     followup_matrix = norm_convert_row_to_prob(followup_matrix)
        # save_followup_matrix_tokens(
        #     followup_matrix_tokens=followup_matrix,
        #     unique_data_id=unique_data_id,
        #     model_tokenization=model_tokenization,
        #     save_images=save_images,
        #     output_folder=output_folder)

        if normalization == "turn_lines_into_prob_distrib":
            model_like_followup_matrix = norm_convert_row_to_prob(model_like_followup_matrix)
        save_followup_matrix_tokens_model(
            followup_matrix_tokens=model_like_followup_matrix,
            unique_data_id=unique_data_id,
            model_tokenization=model_tokenization,
            save_images=save_images,
            output_folder=output_folder)

        if 'line' in levels:

            if normalization == "turn_lines_into_prob_distrib":
                followup_line_matrix = norm_convert_row_to_prob(followup_line_matrix)
            save_followup_matrix_lines(
                followup_matrix_lines=followup_line_matrix,
                unique_data_id=unique_data_id,
                raw_text=metadata["raw_text"],
                save_images=save_images,
                output_folder=output_folder)
            # save_network_lines(
            #     follow_matrix_lines=followup_line_matrix,
            #     unique_data_id=unique_data_id,
            #     raw_text=metadata["raw_text"],
            #     output_folder=output_folder)


@cli.command()
@click.pass_context
@click.option("--username", default=None)
@click.option("--task_number", default=None)
@click.option("--tokenizer", default=None)
@click.option("--override", is_flag=True, default=False)
def getfollowup(ctx, username, tokenizer, task_number, override):
    """Get follow-up attention from sequential events.

    This converts the sequence of eye tracking (line, column) events to a
    relationship strength for each pair of two different tokens.
    The relationship from token A to token B is stronger (high value) if the
    user is often looking at token B after having seen token B.

    Parameters
    ----------
    username : str
        Name of the user.
    task_number : int
        Task number of the given user to inspect.
    tokenizer : str
        Name of the tokenizer to use, it must be a hugging face repository.
        e.g. Salesforce/codegen-16B-multi

    Note that if no username is provided, all participants will be inspected.
    """
    click.echo('Extract follow-up relationship from eye tracker data.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_participants = config['folder_participants']
    dir_questions = config['folder_questions']
    dir_output = config['folder_output']
    participants = config['participants']
    config_followup = config["followup_attention"]
    # the output of the previous step getweigths is the input of this step
    input_folder = os.path.join(dir_output)

    to_extract = []

    if username:
        username = str(username)
        c_user = [
            p for p in participants if p['name'] == username][0]
        if not task_number:
            tasks = c_user['tasks']
            task_number = tasks[0]
        if not tokenizer:
            tokenizer = config["followup_attention"]["model_tokenizer"]
        task_number = int(task_number)
        to_extract.append((username, task_number, tokenizer))
    else:
        click.echo("Extracting all participants data.")
        for p in participants:
            for task_number in p["tasks"]:
                to_extract.append((
                    p['name'], task_number,
                    config["followup_attention"]["model_tokenizer"]))

    for c_decay in config_followup["decays"]:
        for name, task_number, tokenizer in to_extract:
            decay_output_folder = os.path.join(
                dir_output, f"decay_{c_decay}_{tokenizer.replace('/', '_')}")
            pathlib.Path(decay_output_folder).mkdir(parents=True, exist_ok=True)
            click.echo("=" * 80)
            click.echo(
                f"Extracting follow-up attention: {name} (task: {task_number})")
            click.echo(f"Tokenizer: {tokenizer}")
            click.echo(f'Destination folder: {decay_output_folder}')
            # get a unique id for the data by hashing name and task number
            unique_data_id = hashlib.sha256(
                (name + str(task_number)).encode('utf-8')).hexdigest()[:6]
            get_followup_matrix(
                unique_data_id=unique_data_id,
                decay=float(c_decay),
                normalization=config_followup["normalization"],
                tokenizer=tokenizer,
                input_folder=input_folder,
                output_folder=decay_output_folder,
                model_folder=config_followup["model_folder"],
                override_everything=override,
                force_model_tokens=config_followup["force_model_tokens"],
                levels=config_followup["levels"],
                save_images=config_followup["save_images"],
                debug=debug
            )


# =============================================================================
# GET TOKEN TO LINE RELATIONSHIPS
# =============================================================================


def derive_token_to_line_matrix(
        input_folder: str,
        output_folder: str,
        override: bool = False,):
    """Derive the token to line followup attention from the human data.

    This function reads the corresponding data from the followup attention
    folder and derives the token to line followup attention.
    It assumes that the input folder contains a subfolder `metadata` with
    json files with the names corresponding to the npy files in the input
    folder.
    These metadata contain a field for the tokenization, so that it knows to
    which line each token belongs.
    """
    metadata_folder = os.path.join(input_folder, "metadata")
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]
    for file in all_files:
        matrix = np.load(os.path.join(input_folder, file))
        # convert nan to 0
        matrix[np.isnan(matrix)] = 0
        metadata_file = os.path.join(
            metadata_folder, file.replace(".npy", ".json"))
        model_tokenization = json.load(open(metadata_file, "r"))[
            "tokenization"]
        # aggregate new lines which are mapped to the same token index
        line_indices = np.array(
            [t["l"] for t in model_tokenization]
        )[:matrix.shape[1]]
        line_matrix = aggregate_dim_tokens_to_line(
            att_tensor=matrix,
            line_indices=line_indices,
            dim=1)
        # save the matrix if it does not exist yet or if override is True
        output_file = os.path.join(output_folder, file)
        if not os.path.exists(output_file) or override:
            np.save(output_file, line_matrix)



@cli.command()
@click.pass_context
@click.option("--override", is_flag=True, default=False)
def derivefollowupline(ctx, override):
    """Convert the token to token attention to a token-line matrix."""
    click.echo('Extract follow-up relationship from eye tracker data.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_output = config['folder_output']
    config_followup = config["followup_attention"]

    for c_decay in config_followup["decays"]:
        decay_output_folder = os.path.join(
            dir_output, f"decay_{c_decay}")
        pathlib.Path(decay_output_folder).mkdir(parents=True, exist_ok=True)
        click.echo("=" * 80)
        click.echo(
            f"Derive follow-up attention token-line.")
        input_folder = os.path.join(
            decay_output_folder, "data_followup_tokens_tokens_model")
        output_folder = os.path.join(
            decay_output_folder, "data_followup_tokens_lines_model")
        # create folder if it does not exist
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        derive_token_to_line_matrix(
            input_folder=input_folder,
            output_folder=output_folder,
            override=override,
        )


# =============================================================================
# GET BASELINE POSITIONAL ATTENTION
# =============================================================================

@cli.command()
@click.pass_context
def getavgbehavior(ctx):
    """It extract the average behaviour based on the position.

    We human has some content independent behviour (e.g. reading in a linear
    fashior or periodically reading one or two lines above to recall the
    context).
    This function analyze the entire dataset and extract the average
    followup attention given to a token based on its position.
    You need to run getfollowup first.

    It considers all the data in the folder `data_followup_tokens_tokens_model`
    contained in the the decay folder.
    The output is produced in the same decay folder.
    All the decay folders mentioned in the field `followup_attention.decays`
    in the config file are considered .
    """
    click.echo('Extract average follow-up attention.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_output = config['folder_output']

    config_followup = config["followup_attention"]
    tokenizer = config["followup_attention"]["model_tokenizer"]

    for c_decay in config_followup["decays"]:
        decay_folder = os.path.join(
            dir_output, f"decay_{c_decay}_{tokenizer.replace('/', '_')}")
        input_folder = os.path.join(
            decay_folder, "data_followup_tokens_tokens_model")
        output_folder = os.path.join(decay_folder, "data_avg_followup")
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

        human_followup_matrices = read_data_in_parallel(
            base_folder=input_folder,
            file_type_extension=".npy",
            read_function=np.load,)
        all_records = []
        entire_dataset = []
        for k, matrix in tqdm(list(human_followup_matrices.items())[:]):
            for i in range(matrix.shape[0]):
                distrib_to_next_token = matrix[i]
                # get the position of the maximum value
                normalized_distrib = distrib_to_next_token
                if np.sum(distrib_to_next_token) > 0:
                    normalized_distrib = distrib_to_next_token / np.sum(
                        distrib_to_next_token)
                pos_top_1_target = np.argmax(distrib_to_next_token)
                distance_from_max = pos_top_1_target - i
                all_records.append({
                    'unique_data_id': k,
                    'current_token_position': i,
                    'target_position': pos_top_1_target,
                    'prob_to_go_to_top1_target': normalized_distrib[pos_top_1_target],
                    'total_tokens': len(distrib_to_next_token),
                    'distance_current_to_target': distance_from_max,
                })
                for j in range(len(distrib_to_next_token)):
                    entire_dataset.append({
                        'unique_data_id': k,
                        'current_token_position': i,
                        'target_position': j,
                        'total_tokens': len(distrib_to_next_token),
                        'distance_current_to_target': j - i,
                        'prob_to_go_to_target': normalized_distrib[j],
                        'followup_abs_value': distrib_to_next_token[j],
                        'total_followup_abs_value': np.sum(distrib_to_next_token)
                    })
        df_distance_in_tokens = pd.DataFrame.from_records(all_records)
        df_distance_in_tokens.to_csv(
            os.path.join(output_folder, "df_distance_in_from_top1_token.csv"),
            index=False)
        click.echo(f"Saved {os.path.join(output_folder, 'df_distance_in_from_top1_token.csv')}")
        # get the average attention given to a token based on its position
        df_grouped = df_distance_in_tokens.groupby('distance_current_to_target').agg({'prob_to_go_to_top1_target': ['mean', 'count'],})
        df_grouped.columns = ['mean_prob_to_go_to_top1_target', 'number_of_samples']
        df_grouped = df_grouped.reset_index()
        df_grouped.to_csv(
            os.path.join(output_folder, "df_base_top1_prob_based_on_distance.csv"),
            index=False)
        click.echo(
            f"Saved {os.path.join(output_folder, 'df_base_top1_prob_based_on_distance.csv')}")

        # entire dataset
        df_entire_followup = pd.DataFrame.from_records(entire_dataset)
        df_entire_followup.to_csv(
            os.path.join(output_folder, "df_entire_followup.csv"),
            index=False)
        click.echo(f"Saved {os.path.join(output_folder, 'df_entire_followup.csv')}")

        # get the average attention given to a token based on its position
        # group by distance current to target and ge the average probability
        df_grouped = df_entire_followup.groupby('distance_current_to_target').agg({
            "prob_to_go_to_target": ["mean", "count"]})
        # collapse the multiindex
        # FutureWarning: Index.ravel returning ndarray is deprecated; in a future version this will return a view on self.
        df_grouped.columns = df_grouped.columns.to_flat_index()
        # normalize column names
        df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns.values]
        df_grouped = df_grouped.reset_index()
        # invert the distance
        df_grouped['distance_current_to_target'] = df_grouped['distance_current_to_target'] * -1
        # save to file
        df_grouped.to_csv(
            os.path.join(output_folder, "df_base_all_prob_based_on_distance_rebuttal.csv"),
            index=False)
        click.echo(f"Saved {os.path.join(output_folder, 'df_base_all_prob_based_on_distance_rebuttal.csv')}")

        click.echo(f"Decay {c_decay} done.")



# =============================================================================
# GET USERS ANSWERS
# =============================================================================


def get_answer_of(
        unique_data_id: str,
        user_name: str,
        task_number: int,
        dir_participants: str,
        dir_questions: str,
        output_folder: str,
        override_everything: bool = False,
        debug: bool = False):
    """Extract the asnwer for the given user."""
    user_folder = os.path.join(dir_participants, user_name)
    # READ AND SYNCH DATA: EYE AND TEXTUAL DATA
    print(f"Reading data for {user_name} (task {task_number}) ({unique_data_id})")
    try:
        df = read_and_synch_eye_and_text_data(
            user_folder=user_folder,
            task_number=task_number,
            config_manual_tabs={})
    except FileNotFoundError as e:
        print(e)
        return
    # mark the line of the original file which is visible at each timestamp
    df["starting_line"] = df["source_filename"].apply(lambda x:
        int(re.search(r'(\d+)\s$', x).group(1)) if not pd.isna(x) else 0)
    # drop all the rows withput content lines
    df = df[~df["content_lines"].isnull()]
    # count how many visible lines are in the code area
    df["n_lines_visible"] = df["content_lines"].apply(lambda x: len(x))
    df["n_char_visible"] = df["content_lines"].apply(lambda x: len("".join(x)))
    # compute the absolute index of the last line in the code area
    df["last_line_visible"] = df["starting_line"] + df["n_lines_visible"]
    df.sort_values(by=["time"], inplace=True)
    # detect the modified events based on when the numeber of char changes
    df["modified"] = (df["n_char_visible"].diff() != 0) & (df["starting_line"].diff() == 0)
    # detect the modified events based on when the number of lines changes
    # the answer is the text when the last edit is done
    # (assumption the person write an answer from the first line to the
    # last line)

    def get_visible_text(record) -> str:
        """Get the visible text given the timestamp dictionary."""
        return "".join(record["content_lines"])

    def get_answer(record) -> str:
        """Get the answer given the timestamp dictionary."""
        text_answer = get_visible_text(record)
        answer = text_answer[text_answer.index("# Question"):]
        return answer

    target_line = None
    for i in range(len(df) - 1):
        next_record = df.iloc[i + 1]
        current_record = df.iloc[i]
        if current_record["modified"]:
            # consider this modification as the last only if it contains
            # a change next to `# Question` or `# Answer`
            if ("# Question" in "".join(next_record["content_lines"])) and \
                    ("# Answer" in "".join(next_record["content_lines"])):
                target_line = next_record
                #print(next_record["content_lines"])
            if debug:
                print("=" * 80)
                print("Modification detected:")
                print("-" * 80, current_record["time"])
                print(get_visible_text(current_record))
                print("-" * 80, next_record["time"])
                print(get_visible_text(next_record))
    if target_line is None:
        print("No modification detected.")
        answer = ""
        full_filename = df.iloc[len(df) // 2]["source_filename"]
    else:
        # get the question and answer
        answer = "".join(target_line["content_lines"])
        answer = answer[answer.index("# Question"):]
        full_filename = target_line["source_filename"]
    # attach the filename
    filename = re.search(
        r'([a-zA-Z0-9_\.]*):\d+\s$',
        full_filename).group(1)
    answer = filename + "\n" + answer
    # save the answer
    pathlib.Path(os.path.join(output_folder, 'answers')).mkdir(
        parents=True, exist_ok=True)
    output_file = os.path.join(
        output_folder, 'answers', f"{unique_data_id}.txt")
    with open(output_file, "w") as f:
        print(f"Answer >> {answer}")
        f.write(answer)
    print(f"Answer saved to {output_file}")
    return answer


@cli.command()
@click.pass_context
@click.option("--username", default=None)
@click.option("--task_number", default=None)
@click.option("--override", is_flag=True, default=False)
def getanswers(ctx, username, task_number, override):
    """Get the answer of the users in the task."""
    click.echo('Extract the answer from VSCode plugin data.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_participants = config['folder_participants']
    dir_questions = config['folder_questions']
    dir_output = config['folder_output']
    participants = config['participants']

    to_extract = []

    if username:
        username = str(username)
        c_user = [
            p for p in participants if p['name'] == username][0]
        if not task_number:
            tasks = c_user['tasks']
            task_number = tasks[0]
        task_number = int(task_number)
        to_extract.append((username, task_number))
    else:
        click.echo("Extracting all participants data.")
        for p in participants:
            for task_number in p["tasks"]:
                to_extract.append(
                    (p['name'], task_number))

    data_mapping = []
    for i, (name, task_number) in enumerate(to_extract):
        click.echo("=" * 80)
        click.echo(f"Extracting: {name} (task: {task_number})")
        c_dir_participants, c_dir_questions = get_batch_specific_folders(
            config=config,
            usename=name,
            participants=participants,
            dir_participants=dir_participants,
            dir_questions=dir_questions)
        unique_data_id = hashlib.sha256(
            (name + str(task_number)).encode('utf-8')).hexdigest()[:6]
        data_mapping.append({
            "unique_data_id": unique_data_id,
            "user": name,
            "task_number": task_number})
        get_answer_of(
            unique_data_id=unique_data_id,
            user_name=name,
            task_number=task_number,
            dir_participants=c_dir_participants,
            dir_questions=c_dir_questions,
            output_folder=dir_output,
            override_everything=override,
            debug=debug
        )


@cli.command()
@click.pass_context
def getanswersdataset(ctx):
    """Condense the answers in a single dataset."""
    click.echo('Condensing all the answers into a single csv dataset.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_output = config['folder_output']

    dir_answers = os.path.join(dir_output, 'answers')
    output_file = os.path.join(dir_output, 'answers_dataset.csv')

    # load the json file
    with open(os.path.join(dir_output, 'data_mapping.json'), 'r') as f:
        data_mapping = json.load(f)
    records = []
    for task in data_mapping:
        task_answer_txt_path = os.path.join(
            dir_answers, f"{task['unique_data_id']}.txt")
        print("Reading: ", task_answer_txt_path)
        try:
            with open(task_answer_txt_path, 'r') as f:
                task['answer'] = f.read()
        except FileNotFoundError as e:
            print(e)
            continue
        print(task)
        records.append(task)
    df = pd.DataFrame.from_records(records)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    cli()
