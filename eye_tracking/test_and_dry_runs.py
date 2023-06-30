"""This file creates visual inspections of the data for sanity-check."""

import click
from pprint import pprint
import yaml
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import cv2

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
    ctx.obj['debug'] = debug


def compute_visual_angle(
        screen_distance_in_mm: float,
        object_size_in_mm: float) -> float:
    """Compute the visual angle of the screen."""
    arctan_in_rad = np.arctan(object_size_in_mm / (2 * screen_distance_in_mm))
    return np.rad2deg(arctan_in_rad) * 2


def compute_visual_angle_in_pixel(
        screen_distance_in_mm: float,
        object_size_in_pixel: float,
        screen_size_in_pixel: float,
        screen_size_in_mm: float) -> float:
    """Compute the visual angle of the screen (given a object size in pixel)."""
    object_size_in_mm = \
        (object_size_in_pixel / screen_size_in_pixel) * screen_size_in_mm
    return compute_visual_angle(screen_distance_in_mm, object_size_in_mm)


@cli.command()
@click.pass_context
@click.option("--figsize", default=10)
@click.option("--visual_region_size", default=5)
@click.option("--override", is_flag=True, default=False)
def inspectgrids(ctx, figsize, visual_region_size, override):
    """Inspects the grid for each task."""
    click.echo('Extract attention weights from eye tracker data.')
    debug = ctx.obj.get('DEBUG', False)
    config = ctx.obj['CONFIG']
    dir_participants = config['folder_participants']
    dir_output = config['folder_output']
    participants = config['participants']

    # crate a dry-run folder in the output folder if it does not exist yet
    path_dry_run = os.path.join(dir_output, "dry_run_grid")
    pathlib.Path(path_dry_run).mkdir(parents=True, exist_ok=True)

    # for each participants
    for participant in participants:
        participant_name = participant['name']
        if "batches" in config.keys():
            c_batch_no = participant["batch_no"]
            c_batch_info = [
                b for b in config['batches'] if b["batch_no"] == c_batch_no][0]
            dir_participants = c_batch_info["folder_participants"]
        click.echo("=" * 80)
        click.echo(f"Scraping participant: {participant_name}")
        # get the path of the first avi file of the participant
        path_participant = pathlib.Path(
            os.path.join(dir_participants, participant_name))
        try:
            click.echo("Considering only one video.")
            video_path = list(sorted(path_participant.glob("*.avi")))[1]
        except IndexError:
            click.echo(f"Skipping video: {path_participant} because of lack of video footage.")
        click.echo(video_path)
        # get video data
        video_data = get_tot_n_frames(video_path)
        mid_frame = int(video_data["tot_n_frames"] / 2)
        click.echo(f"Mid frame: {mid_frame}")
        # get the screenshot of the first frame of the video
        img, screen_path = get_video_image(
            video_path=str(video_path),
            frame_number=mid_frame,
            output_folder=path_dry_run)

        zoom_level = participant['zoom']
        params = config["zoom_dependant_parameters"][zoom_level]

        manual_screen_coordinates = participant.get(
            "manual_screen_coordinates", None)

        code_screen_coordinate_abs = get_code_coordinates(
            screenshot_path=str(screen_path),
            path_marker_top_left=params["path_marker_top_left"],
            path_marker_bottom_right=params["path_marker_bottom_right"],
            top_left_offsets=params["top_left_offsets"],
            bottom_right_offsets=params["bottom_right_offsets"],
            percentage_format=False,
            debug=False)

        if manual_screen_coordinates:
            print("Using manual screen cooridnates...")
            code_screen_coordinate_abs = manual_screen_coordinates

        click.echo("Code screen coordinate absolute: ")
        pprint(code_screen_coordinate_abs)

        # COMPUTE THE CHAR SPAN
        n_lines = params["n_lines"]
        n_col = params["n_col"]

        x_start_screen_abs = code_screen_coordinate_abs["x_start_screen_abs"]
        y_start_screen_abs = code_screen_coordinate_abs["y_start_screen_abs"]
        x_end_screen_abs = code_screen_coordinate_abs["x_end_screen_abs"]
        y_end_screen_abs = code_screen_coordinate_abs["y_end_screen_abs"]

        code_area_width = x_end_screen_abs - x_start_screen_abs
        code_area_height = y_end_screen_abs - y_start_screen_abs
        char_width = code_area_width / n_col
        char_height = code_area_height / n_lines
        click.echo(f"Char size: ({char_width}, {char_height})")

        # Compute in viewing angle
        char_width_viz_angle = compute_visual_angle_in_pixel(
            screen_distance_in_mm=config["screen_distance_mm"],
            object_size_in_pixel=char_width,
            screen_size_in_pixel=config["screen_resolution"]["pixel_screen_width"],
            screen_size_in_mm=config["screen_size"]["width_mm"])
        char_height_viz_angle = compute_visual_angle_in_pixel(
            screen_distance_in_mm=config["screen_distance_mm"],
            object_size_in_pixel=char_height,
            screen_size_in_pixel=config["screen_resolution"]["pixel_screen_height"],
            screen_size_in_mm=config["screen_size"]["height_mm"])
        click.echo(f"Char size (visual angle): ({char_width_viz_angle}, {char_height_viz_angle})")

        click.echo(f"In our region of {visual_region_size} degrees, we can see:")
        n_horizontal_chars = visual_region_size / char_width_viz_angle
        click.echo(f"{n_horizontal_chars:.2f} horizontal chars")
        n_vertical_chars = visual_region_size / char_height_viz_angle
        click.echo(f"{n_vertical_chars:.2f} vertical chars")

        col_separators = [
            x_start_screen_abs + char_width * i
            for i in range(n_col + 1)
        ]
        line_separators = [
            y_start_screen_abs + char_height * i
            for i in range(n_lines + 1)
        ]
        fig, ax = plt.subplots(1, figsize=(figsize, figsize))
        ax.imshow(img)
        for col_sep in col_separators:
            ax.axvline(col_sep, color='r', linewidth=1)
        for line_sep in line_separators:
            ax.axhline(line_sep, color='r', linewidth=1)

        path_screen_with_grid = os.path.join(
            path_dry_run, f"{participant_name}_grid.png")
        fig.savefig(path_screen_with_grid)
        click.echo(f"Saved grid to {path_screen_with_grid}")


if __name__ == '__main__':
    cli()
