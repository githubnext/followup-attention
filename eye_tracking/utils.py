
from collections import Counter
from venv import main
import cv2
import os
import numpy as np
from os.path import join
from pathlib import Path
import re
import json
from typing import List, Tuple, Any, Union, Dict
import pandas as pd
import imutils
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm

import tempfile

import seaborn as sns

import yaml
from datetime import timedelta
from datetime import datetime

import ffmpeg
from matplotlib import font_manager

import pandas as pd


SIZE = 20

# VIDEO RELATED


def get_video_image(
        video_path: str,
        frame_number: int,
        output_folder: str = None):
    """
    Get the image from the video at the given frame number.
    """
    if not os.path.exists(video_path):
        print("Path does not exist:", video_path)
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, image = video.read()
    # print("Return value: ", ret)
    if not ret:
        raise Exception("Could not read image at frame number: ", frame_number)
    # get participant name from video path
    image_path = None
    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        image_path = join(output_folder, f"frame_{frame_number}.jpg")
        cv2.imwrite(image_path, image)
    return image, image_path


def get_first_white_screen(
        video_path: str,
        output_folder: str = None,
        whiteness_threshold: str = 180,
        start_frame: int = 0,
        scan_every_n_frames: int = 50):
    """Iterate over frames until we find a screen which is mostly white."""
    white_screen_found = False
    c_frame = start_frame
    while not white_screen_found:
        c_frame += scan_every_n_frames
        img, _ = get_video_image(
            video_path=str(video_path),
            frame_number=c_frame)
        # get the average value of a pixel in the screenshot
        avg_value = np.average(img)
        if avg_value > whiteness_threshold:
            white_screen_found = True
    return get_video_image(
        video_path=str(video_path),
        frame_number=c_frame,
        output_folder=output_folder)


def cut_image(
        image_array,
        x_start_cut: int = 0,
        y_start_cut: int = 0,
        x_end_cut: int = None,
        y_end_cut: int = None):
    """Cut an image."""
    if x_end_cut is None:
        x_end_cut = image_array.shape[1]
    if y_end_cut is None:
        y_end_cut = image_array.shape[0]
    # convert all to integer values
    x_start_cut = int(x_start_cut)
    x_end_cut = int(x_end_cut)
    y_start_cut = int(y_start_cut)
    y_end_cut = int(y_end_cut)
    return image_array[y_start_cut:y_end_cut, x_start_cut:x_end_cut, :]


# COMPUTER VISION - DETECTION / TEMPLATE MATCHING


def get_coordinate_of(marker_path: str, screen_path: str, debug: bool = False):
    """Locate the coordinate of the marker in the given image."""
    # load marker image
    template = cv2.imread(marker_path)
    # load screen image
    image = cv2.imread(screen_path)
    return get_coordinate_of_array(template, image, debug=debug)


def get_coordinate_of_array(template, image, debug: bool = False):
    """Locate the coordinate of the marker in the image array.

    Both marker and image are already passed as arrays.
    Thus no I/O is performed by this function."""

    # convert both the image and template to grayscale
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    if debug:
        plt.imshow(template)
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # check to see if the iteration should be visualized
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # draw a bounding box around the detected result and display the image
    # plt.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    if debug:
        fig, ax = plt.subplots(1, figsize=(SIZE, SIZE))
        ax.imshow(image)
        rect = patches.Rectangle(
            (startX, startY), endX-startX, endY-startY,
            linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()
    return startX, startY, endX, endY


def get_code_coordinates(
        screenshot_path: str,
        path_marker_top_left: str = "../markers/top_left_marker.png",
        path_marker_bottom_right: str = "../markers/bottom_right_marker.png",
        top_left_offsets: Tuple[int, int] = (182, 109),
        bottom_right_offsets: Tuple[int, int] = (107, 6),
        percentage_format: bool = False,
        debug: bool = False):
    """Find the screen markers in the screenshot.

    It returns the start and end x and y coordinate of the screen.
    It can be returned as percentage of the screen size or in pixels, as decided
    by the percentage_format parameter which is False by default.

    Example of output:
    # percentage format
        {
            "x_start_screen_perc": 0.3,
            "y_start_screen_perc": 0.1,
            "x_end_screen_perc": 0.8,
            "y_end_screen_perc": 0.9
        }

    # absolute format
        {
            "x_start_screen_abs": 576,
            "y_start_screen_abs": 108,
            "x_end_screen_abs": 1536,
            "y_end_screen_abs": 972
        }
    """

    # cut to consider only the upper right corner
    img = cv2.imread(screenshot_path)
    img_height, img_width = img.shape[:2]
    img_cut = cut_image(img, 0, 0, img_width * .3, img_height*.2)
    # save to file
    screenshot_path_cut = screenshot_path.replace(".jpg", "_cut.jpg")
    cv2.imwrite(screenshot_path_cut, img_cut)

    # get top-left screen coordinates
    startX, startY, endX, endY = get_coordinate_of(
        marker_path=path_marker_top_left,
        screen_path=screenshot_path_cut,
        debug=debug
    )

    horizontal_screen_start = startX + top_left_offsets[0]
    vertical_screen_start = startY + top_left_offsets[1]

    # get bottom-right screen coordinates

    startX, startY, endX, endY = get_coordinate_of(
        marker_path=path_marker_bottom_right,
        screen_path=screenshot_path,
        debug=debug
    )

    horizontal_screen_end = startX + bottom_right_offsets[0]
    vertical_screen_end = startY + bottom_right_offsets[1]

    # visualize result
    if debug:
        fig, ax = plt.subplots(1, figsize=(SIZE, SIZE))
        image = cv2.imread(screenshot_path)
        ax.imshow(image)

        ax.axvline(horizontal_screen_start, color='r', linewidth=1)
        ax.axhline(vertical_screen_start, color='r', linewidth=1)

        ax.axvline(horizontal_screen_end, color='r', linewidth=1)
        ax.axhline(vertical_screen_end, color='r', linewidth=1)
        plt.show()

    if percentage_format:
        return {
            "x_start_screen_perc": horizontal_screen_start / image.shape[1],
            "y_start_screen_perc": vertical_screen_start / image.shape[0],
            "x_end_screen_perc": horizontal_screen_end / image.shape[1],
            "y_end_screen_perc": vertical_screen_end / image.shape[0]
        }

    return {
        "x_start_screen_abs": horizontal_screen_start,
        "y_start_screen_abs": vertical_screen_start,
        "x_end_screen_abs": horizontal_screen_end,
        "y_end_screen_abs": vertical_screen_end
    }


def get_char_coordinates(
        eye_x_perc, eye_y_perc,
        x_start_screen_abs: float,
        y_start_screen_abs: float,
        x_end_screen_abs: float,
        y_end_screen_abs: float,
        n_lines: int = 26,
        n_col: int = 97,
        screen_path: str = None,
        pixel_screen_width: int = None, pixel_screen_height: int = None,
        ax=None,
        debug: bool = True
        ) -> Tuple[int, int]:
    """It converts a screen percentage to a specific (line, column) coordinate.

    Remember to provide either the screen_path or the pixel width and height,
    otherwise we have no way to figure out how large the screen is.
    Note that the coordinate system starts from 0 for both lines and columns.
    Thus the top left character of the code screen is (0, 0).

    Example output:
        (4, 7) meaning the char on the 5th line and the 8th column
    """
    image = None
    if pixel_screen_width is None or pixel_screen_height is None:
        image = cv2.imread(screen_path)
        pixel_screen_width = image.shape[1]
        pixel_screen_height = image.shape[0]
    code_area_width = x_end_screen_abs - x_start_screen_abs
    code_area_height = y_end_screen_abs - y_start_screen_abs
    char_width = code_area_width / n_col
    char_height = code_area_height / n_lines

    col_separators = [
        x_start_screen_abs + char_width * i
        for i in range(n_col + 1)
    ]
    line_separators = [
        y_start_screen_abs + char_height * i
        for i in range(n_lines + 1)
    ]

    # convert eye cursor in absolute
    eye_x_abs = eye_x_perc * pixel_screen_width
    eye_y_abs = eye_y_perc * pixel_screen_height

    # get the column index
    col_index = 0
    for i in range(len(col_separators) - 1):
        if col_separators[i] <= eye_x_abs < col_separators[i + 1]:
            col_index = i
            break
    # get the line index
    line_index = 0
    for i in range(len(line_separators) - 1):
        if line_separators[i] <= eye_y_abs < line_separators[i + 1]:
            line_index = i
            break

    if debug:
        # visualize
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(SIZE, SIZE))
        if not image:
            image = cv2.imread(screen_path)
        ax.imshow(image)
        for col_sep in col_separators:
            ax.axvline(col_sep, color='r', linewidth=1)
        for line_sep in line_separators:
            ax.axhline(line_sep, color='r', linewidth=1)
        # add a point in the eye cursor
        ax.scatter(eye_x_abs, eye_y_abs, c='red', s=100)
        plt.show()
    # print(f"({line_index}, {col_index})")
    return (line_index, col_index)


def convert_to_absolute(perc_coordinate: float, total_space: int):
    """Convert a percentage coordinate to an absolute coordinate."""
    return int(perc_coordinate * total_space)


# Data Processing - iTracker (VSCode plugin)


def convert_string_to_day_time(snapshot_filename: str) -> int:
    """Convert a snapshot string to the number of millisecond of that day."""
    # remove file extension
    time_string = snapshot_filename.split(".")[0]
    # in case there are duplicated filenames:
    # e.g. 2-04-56-PM-975.txt and 2-04-56-PM-975(1).txt
    # because there are multiple scrolls within the same millisecond, then
    # add 1 millisecond to the more recent file
    time_string = time_string.replace("(", "").replace(")", "")
    # get the date and time of the snapshot
    date_time = datetime.strptime(
        time_string,
        "%I-%M-%S-%p-%f"
    )
    # print(date_time)
    # get the number of milliseconds seconds in date time
    return (date_time - datetime(1900, 1, 1)).total_seconds() * 1000


def fix_tab_problems(line, config_manual_tabs: List[Dict[str, Any]]):
    """Substitute tab space to uniform to the vscode format.

    This happens because vscode shows tab in different ways depending on the
    language, thus we replace all the ambiguous tab space with the equivalent
    number of spaces as those seen by the user during the experiment.

    Note that this was present only in the first batch of user, since
    after we used only spaces in our source code file.
    """
    for replace_unit in config_manual_tabs:
        line = line.replace(
            replace_unit["find"],
            replace_unit["replace"])
    return line


def read_textual_snapshot(
        user_folder: str,
        config_manual_tabs: List[Dict[str, Any]] = None):
    """Read the textual snapshot of the code screen."""
    records = []
    # if the record.json exists read that
    record_path = join(user_folder, "Snapshots", "record.json")
    if os.path.exists(record_path):
        with open(record_path, "r") as f:
            records = json.load(f)
        return pd.DataFrame.from_records(records)
    if not config_manual_tabs:
        config_manual_tabs = []
    for snapshot_filename in os.listdir(join(user_folder, "Snapshots")):
        file = open(join(user_folder, "Snapshots", snapshot_filename), "r")
        lines = file.readlines()
        record = {
            "time": convert_string_to_day_time(snapshot_filename),
            "source_filename": lines[0],
            "content_lines": [
                fix_tab_problems(l, config_manual_tabs).replace("\t", "    ")
                for l in lines[1:]]
        }
        records.append(record)
    # dump records to a json file
    with open(record_path, "w") as f:
        json.dump(records, f)
    return pd.DataFrame.from_records(records)


def get_textual_snapshot_at(df_snapshots: pd.DataFrame, time_ms: int):
    """Get the textual snapshot at a given time."""
    df_snapshots = df_snapshots.sort_values(by="time")
    df_previous_states = df_snapshots.loc[df_snapshots["time"] < time_ms]
    if len(df_previous_states) == 0:
        raise Exception("No snapshot found before the start time.")
    # get last row
    last_row = df_previous_states.iloc[-1].to_dict()
    return last_row


def display_text(
        content: str,
        font_path: str = "../consolas/vscode_windows_font.TTF",
        font_size: int = 20,
        debug: bool = False,
        show_line_numbers: bool = False,
        ax=None):
    """Display content in matplotlib with the given font."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    if show_line_numbers:
        lines = content.splitlines()
        numbered_lines = []
        for i, line in enumerate(lines):
            new_line = str(i + 1).zfill(2) + "| " + line
            numbered_lines.append(new_line)
        content = "\n".join(numbered_lines)
    prop = font_manager.FontProperties(fname=font_path)
    text_obj = ax.text(0, 0, content, fontsize=font_size, fontproperties=prop)
    if not debug:
        ax.axis("off")
    return ax


# Data Processing - OBS Studio


def get_tot_n_frames(video_path: str):
    """Get the total number of frames."""

    data = ffmpeg.probe(video_path)["streams"][0]
    return {
        "tot_n_frames": data["duration_ts"],
        "fps": data["avg_frame_rate"].replace("/1", "")
    }


def get_video_path(user_folder: str, task_nr: int) -> str:
    """Get the path of the video file."""
    all_video_paths = [
        (
            int(re.search(r'([0-9])+.avi', Path(v).name).group(1)),
            v
        )
        for v in os.listdir(user_folder)
        if v.endswith(".avi")
    ]
    target_video_path = [
        v for nr, v in all_video_paths
        if nr == task_nr
    ][0]
    return join(user_folder, target_video_path)


# Data Processing - Eye Tracking Data


def offset_time_based_on_time_column(df: pd.DataFrame):
    """Offset eye tracker data based on the offset in the time column."""
    time_col = [c for c in df.columns if "TIME(" in c][0]
    #print(time_col)
    # read the time stamp in the time_col column
    time_string = time_col.split(" ")[1]
    # get the date and time of the snapshot
    date_time = datetime.strptime(
        time_string,
        "%H:%M:%S.%f)"
    )
    # print(date_time)
    # get the number of milliseconds seconds in date time
    offset = (date_time - datetime(1900, 1, 1)).total_seconds() * 1000
    df["time_abs"] = df[time_col] * 1000 + offset
    df["FPOGS_abs"] = df["FPOGS"] * 1000 + offset
    return df


def read_eye_tracker_fixations(user_folder: str, task_nr: int):
    """Reads eye tracker fixation for the given user."""
    # get all csv files in the user folder
    csv_files = [
        (
            int(re.search(r'([0-9])+_fixations.csv$', Path(f).name).group(1)),
            Path(f).name
        )
        for f in os.listdir(user_folder)
        if f.endswith(".csv") and "_fixations" in f]
    # keep only fixations
    #csv_files = [f for f in csv_files if "fixations" in f[1]]
    # get the path of the csv file for the given task
    csv_file = [f for f in csv_files if f[0] == task_nr][0][1]
    # read the csv file
    df = pd.read_csv(join(user_folder, csv_file))
    return df


def get_eye_tracker_fixations_at(
        df: pd.DataFrame, time_ms: int, time_col: str = "FPOGS_abs"):
    """Get the eye tracker fixations at a given time."""
    df = df.sort_values(by=time_col)
    df_previous_states = df.loc[df[time_col] < time_ms]
    # get last row
    last_row = df_previous_states.iloc[-1].to_dict()
    return last_row


# ANNOTATIONS


def get_time_annotation_per_task(
        user_name: str,
        task_nr: int,
        config_folder: str = "../config"):
    """Read the yaml file containing the annotation on the video timestamps."""
    # read annotation
    annotation_path = join(
        config_folder, f"annotation_{user_name}_{task_nr}.yaml")
    print(annotation_path)
    with open(annotation_path, "r") as f:
        annotation = yaml.safe_load(f)
    # parse the dates
    start_task_rel = datetime.strptime(annotation["task_start"],"%M:%S")
    end_task_rel = datetime.strptime(annotation["task_end"],"%M:%S")
    synch_rel = datetime.strptime(annotation["sync_timestamp_video"],"%M:%S")
    synch_abs = datetime.strptime(annotation["sync_timestamp_pc_clock"],"%H:%M")
    positive_offset = synch_abs - synch_rel
    start_task_abs = positive_offset + start_task_rel
    end_task_abs = positive_offset + end_task_rel
    annotation["positive_offset_ts_ms"] = int(positive_offset.total_seconds()) * 1000
    annotation["start_task_abs_ts_ms"] = \
        (start_task_abs - datetime(1900, 1, 1)).total_seconds() * 1000
    annotation["end_task_abs_ts_ms"] = \
        (end_task_abs - datetime(1900, 1, 1)).total_seconds() * 1000
    return annotation


# VISUALIZATION

def get_x_y_pos_of_eye(
        video_path: str,
        marker_eye_tracker_dot: str = "../markers/pointer.png",
        stop_after_n_frames: int = None,
        debug: bool = False):
    """Create a dataframe with x, y position of the marker over time."""
    data = get_tot_n_frames(video_path)
    total_frames = data["tot_n_frames"]
    template = cv2.imread(marker_eye_tracker_dot)

    records = []
    if (stop_after_n_frames is not None and
            total_frames > stop_after_n_frames):
        total_frames = stop_after_n_frames

    for i in tqdm(range(0, total_frames)):
        img, _ = get_video_image(
            video_path=video_path,
            frame_number=i
        )
        # to avoid to match the magnifiying glass on the sidebar of VSCode
        # we cut the image
        shift_for_cut_in_px = 200
        img = cut_image(img, x_start_cut=shift_for_cut_in_px)

        startX, startY, endX, endY = get_coordinate_of_array(
            template=template,
            image=img,
            debug=debug
        )
        if startX is not None:
            new_record = {
                "x": (startX + endX) / 2 + shift_for_cut_in_px,
                "y": (startY + endY) / 2,
                "frame_number": i,
            }
            records.append(new_record)
    df = pd.DataFrame.from_records(records)
    return df


def inspect_x_y_position(
        df: pd.DataFrame,
        time_col: str = "FPOGS",
        x_col: str = "CX",
        y_col: str = "CY",
        vertical_ax_label: str = "\% of screen size"):
    """Inspect the x and y position of the mouse cursor."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_x_pos = axes[0]
    ax_y_pos = axes[1]
    sns.lineplot(
        data=df,
        x=time_col,
        y=x_col,
        ax=ax_x_pos
    )
    ax_x_pos.set_title("X position")
    ax_x_pos.set_xlabel("Time")
    ax_x_pos.set_ylabel(vertical_ax_label)
    ax_x_pos.axhline(1, color="red", linestyle="--")
    sns.lineplot(
        data=df,
        x=time_col,
        y=y_col,
        ax=ax_y_pos
    )
    ax_y_pos.set_title("Y position")
    ax_y_pos.set_xlabel("Time")
    ax_y_pos.set_ylabel(vertical_ax_label)
    ax_y_pos.axhline(1, color="red", linestyle="--")
    # compute the last timestam after which
    # all the data are between zero and one
    plt.show()


def inspect_video_vs_code_text(
        user_folder: str,
        user_zoom: str,
        task_nr: int,
        every_n_millisec: int = 1000,
        start_millisec: int = 120000,
        n_samples: int = 5):
    """Inspect the video vs the iTracker data every 5 seconds."""
    # reason on millisec

    video_path = get_video_path(user_folder, task_nr)
    fps = int(get_tot_n_frames(video_path)["fps"])
    # get video offset in ms
    annotation = get_time_annotation_per_task(
        user_name=user_folder.split("/")[-1],
        task_nr=task_nr)
    positive_offset_ts_ms = annotation["positive_offset_ts_ms"]

    # read the eye tracker data
    df_snap = read_textual_snapshot(user_folder)
    earliest_itracker_data_ts_ms  = df_snap["time"].min()
    print("Earliest Eye-Tracker Data (millisec): ", earliest_itracker_data_ts_ms)

    # check if the eye tracker has enough data
    if earliest_itracker_data_ts_ms > start_millisec + positive_offset_ts_ms:
        raise Exception(
            f"The eye tracker data is not available before the participants"
            "opens a file. Try increasing the start_second value. "
        )


    for i in range(start_millisec, start_millisec + every_n_millisec * n_samples, every_n_millisec):
        fig, axes = plt.subplots(1, 3, figsize=(30, 5.5))
        ax_video = axes[0]
        ax_itracker = axes[1]
        ax_eye = axes[2]

        # get video frame
        video_time_millisec = i
        video_time_frame = round(video_time_millisec / 1000 * fps)

        tmp_dir = tempfile.TemporaryDirectory(dir="./tmp")
        img, img_path = get_video_image(
            video_path=video_path,
            frame_number=video_time_frame,
            output_folder=tmp_dir.name)
        ax_video.imshow(img)
        ax_video.set_title(f"Video (frame: {video_time_frame}, sec: {video_time_millisec/1000})")

        # get the itracker data
        itracker_time_millisec = video_time_millisec + positive_offset_ts_ms + int(annotation["shift_text_ms"])
        record = get_textual_snapshot_at(df_snap, time_ms=itracker_time_millisec)
        display_text(
            content="".join(record["content_lines"]).replace("\t", "    "),
            font_size=10,
            debug=False,
            ax=ax_itracker,
        )
        ax_itracker.set_title(f"iTracker ({itracker_time_millisec} ms)")

        # get the eye-tracker data
        eye_tracker_fixation_time_millisec = video_time_millisec + positive_offset_ts_ms + int(annotation["shift_eye_data_ms"])
        df_eye = read_eye_tracker_fixations(
            user_folder=user_folder,
            task_nr=task_nr,
        )
        df_eye = offset_time_based_on_time_column(df_eye)

        current_fixation = get_eye_tracker_fixations_at(df_eye,
            time_ms=eye_tracker_fixation_time_millisec)
        fixation_X = current_fixation["FPOGX"]
        fixation_Y = current_fixation["FPOGY"]

        if user_zoom == "M":
            marker_params = {
                "path_marker_top_left": "../markers/top_left_marker.png",
                "path_marker_bottom_right": "../markers/bottom_right_marker.png",
                "top_left_offsets": (182, 109),
                "bottom_right_offsets": (107, 6),
            }
            col_lines_params = {"n_lines": 26, "n_col": 97,}
        elif user_zoom == "L":
            marker_params = {
                "path_marker_top_left": "../markers/top_left_marker_zoomed.png",
                "path_marker_bottom_right": "../markers/bottom_right_marker_zoomed.png",
                "top_left_offsets": (220, 125),
                "bottom_right_offsets": (90, -25),
            }
            col_lines_params = {"n_lines": 20, "n_col": 71,}

        code_screen_coordinate_abs = get_code_coordinates(
            screenshot_path=img_path,
            percentage_format=False,
            debug=False,
            **marker_params)

        ax_eye.set_title(f"Eye-Tracker ({eye_tracker_fixation_time_millisec} ms)")
        get_char_coordinates(
            eye_x_perc=fixation_X, eye_y_perc=fixation_Y,
            pixel_screen_width=1920, pixel_screen_height=1080,
            screen_path=img_path,
            ax=ax_eye,
            debug=True,
            **col_lines_params,
            **code_screen_coordinate_abs)
        tmp_dir.cleanup()


# AutoSynchronization


def is_perfect_match(
        video_path: str,
        content: str,
        video_ts_ms: int,
        debug: bool = False,):
    """Check if the code matches the video image."""
    tmp_dir = tempfile.TemporaryDirectory(dir="./tmp")

    fps = int(get_tot_n_frames(video_path)["fps"])
    video_time_frame = round(video_ts_ms / 1000 * fps)
    # extract the video frame
    img, img_path =get_video_image(
        video_path=video_path,
        frame_number=video_time_frame,
        output_folder=tmp_dir.name)
    # get the itracker data and create text image

    fig, ax_itracker = plt.subplots(1, 1, figsize=(10, 5))
    display_text(
        content=content,
        font_size=20,
        debug=False,
        ax=ax_itracker,
    )
    plt.tight_layout()
    print("tmp_dir.name: ", tmp_dir.name)
    marker_path = join(str(tmp_dir.name), "synch_text.jpg")
    fig.savefig(marker_path)
    plt.close()


    # match the text in the video frame
    coord_of_text = get_coordinate_of(
        marker_path=marker_path,
        screen_path=img_path,
        debug=debug,
    )

    tmp_dir.cleanup()
    return coord_of_text


def get_neighbors_snapshots(
        df: pd.DataFrame,
        anchor_ts_ms: int,
        col_time: str = "time",
        col_content: str = "content_lines",
        n_tot_snapshots: int = 10,
        distance_between_snapshots: int = 20):
    """Get the snapshots in the surrounding of the current snapshot.

    Half of the value of n_tot_snapshots decides how many snapshots to include
    with a timestamp lower or equal than the anchor timestamp, and the same
    parameter decides also how many snapshots to include with a timestamp higher
    than the snapshot. Note that in the end the interval is equal.

    Returns:
     [
        (ts_ms_1, content_1),
        ...
        (ts_ms_10, content_10),
     ]
    """
    df = df.copy().sort_values(by=col_time)
    df = df.iloc[::distance_between_snapshots, :]
    n_before = n_after = n_tot_snapshots // 2
    df_before = df[df[col_time] <= anchor_ts_ms].iloc[-n_before:]
    df_after = df[df[col_time] > anchor_ts_ms].iloc[:n_after]
    df_closest = pd.concat([df_before, df_after])
    records = df_closest.to_dict("records")
    return [
        (r[col_time], "".join(r[col_content]).replace("\t", "    "))
        for r in records]


def find_closest_snapshot_ts(
        video_path: str,
        video_ts_ms: int,
        df_text_snapshots: pd.DataFrame,
        guess_ts_ms: int,
        x_code_coord: int,
        n_checks: int = 10,
        distance_between_snapshots: int = 20,
        debug: bool = False,):
    """Given a timeframe find the closest snapshot, it returns its timestamp.

    The parameter guess_ts_ms give a hint on where to look for the correct
    snapshot, beside that precise snapshot, it will check the snapshots right
    before and right after that one. How many snapshots to check is decided by
    the n_checks, half checks before and half checks after the current snapshot.
    """
    closest_snapshots = get_neighbors_snapshots(
        df_text_snapshots,
        anchor_ts_ms=guess_ts_ms,
        n_tot_snapshots=n_checks,
        distance_between_snapshots=distance_between_snapshots,
    )
    candidates = []
    for candidate_ts, content in closest_snapshots:
        res = is_perfect_match(
            video_path=video_path,
            content=content,
            video_ts_ms=video_ts_ms,
            debug=debug,
        )
        if res is not None:
            candidates.append((candidate_ts, res))

    candidates = [
        c for c in candidates
        if c[1][0] >= x_code_coord]
    return candidates[0][0]


# Data Processing - Attention Extraction


def convert_char_coordinate_to_attention_area(
        line: int,
        col: int,
        content: str,
        content_starting_line: int,
        horizontal_attention_span: int = 6,
        vertical_attention_span: int = 1,
        tot_screen_lines: int = 26,
        tot_screen_col: int = 97,
        debug: bool = False):
    """It converts (line, col) coordinates to the absolute file positions.

    Important: Note that line and col should start from 1!
    Also the output is given with the same convention.
    Whereas the content_starting_line starts with 0.

    It returns a set of character coordinates which have been attended.
    Note that the horizontal_attention_span and vertical_attention_span define
    which other characters in the vicinity of the main character are also
    attended.
    """
    # convert line and col to absolute file positions
    line += content_starting_line
    # get the character coordinates of the neighborhood of the main character
    neighborhood_char_coords = []
    if debug:
        content_lines = content.split("\n")
        padded_lines = []
        for i, line_content in enumerate(content_lines):
            #print(str(i+1).zfill(3) + ") " + line_content)
            padded_lines.append(line_content + " " * (tot_screen_col - len(line_content)))
        # add padding lines
        for i in range(tot_screen_lines):
            padded_lines.append(" " * tot_screen_col)
        #print(f"MAIN CHARACTER: ({line}, {col}): {padded_lines[line - content_starting_line][col]}")
    for i in range(line - vertical_attention_span, line + vertical_attention_span + 1):
        is_out_of_screen_vertically = (
            i < content_starting_line + 1 or
            i > content_starting_line + tot_screen_lines)
        if is_out_of_screen_vertically:
            continue
        for j in range(col - horizontal_attention_span, col + horizontal_attention_span + 1):
            is_out_of_screen_horizontally = (j < 1 or j > tot_screen_col)
            if is_out_of_screen_horizontally:
                continue
            neighborhood_char_coords.append((i, j))
            if debug:
                target_char = padded_lines[i - content_starting_line][j]
                #print(f"i, j = {max(0, i - content_starting_line) + 1}, {j} - > `{target_char}`")
                print(target_char, end="")
        if debug:
            print("")
    # print(f"content_starting_line: {content_starting_line}")
    # print("content: " + content)
    assert all(
        (i >= 1 and j >= 1) for i, j in neighborhood_char_coords), \
        "Neighborhood character coordinates should be greater than 1 " + \
        f"({line}, {col}) -> {neighborhood_char_coords}"
    max_neighborhood_size = \
        (horizontal_attention_span * 2 + 1) * (vertical_attention_span * 2 + 1)
    assert len(neighborhood_char_coords) <= max_neighborhood_size, \
        "Neighborhood character coordinates must be less than or equal to " + \
        f"max_neighborhood_size ({max_neighborhood_size}) = " + \
        "(horizontal_attention_span * 2 + 1) * (vertical_attention_span * 2 + 1)"
    return neighborhood_char_coords


def extract_attention_map_char_level(
        user_folder: str,
        user_zoom: str,
        task_nr: int,
        every_n_millisec: int = 250,
        start_millisec: int = None,
        n_samples: int = -1,
        horizontal_attention_span: int = 6,
        vertical_attention_span: int = 1,
        assume_code_area_is_not_resized: bool = True,
        debug: bool = True) -> Tuple[str, List[Tuple[int, int]]]:
    """Derive the attention map at a character level.

    This script requires an annotation file with information about:
    - task_start: "1:40"
    - task_end: "4:37"
    - sync_timestamp_pc_clock: "14:41"
    - sync_timestamp_video: "1:31"
    - shift_text_ms: 0
    - shift_eye_data_ms: -400

    Note that n_samples is -1 it will sample at the given interval of
    every_n_millisec until the end of the task.

    Example output:
    - relative filepath of the source code file of the task, relative w.r.t.
        the task folder
    - List of tuple with (line, col) which were visible for a fraction of the
        time (every_n_millisec). Note that the output might contain the same
        (line, col) coordinate multiple times.
    """
    # reason on millisec
    video_path = get_video_path(user_folder, task_nr)
    print(f"video: {video_path}")
    fps = int(get_tot_n_frames(video_path)["fps"])
    # get video offset in ms
    annotation = get_time_annotation_per_task(
        user_name=user_folder.split("/")[-1],
        task_nr=task_nr)
    positive_offset_ts_ms = annotation["positive_offset_ts_ms"]
    end_task_in_video_millisec = \
        annotation["end_task_abs_ts_ms"] - positive_offset_ts_ms

    # handle default settings without start_millisec
    if start_millisec is None:
        start_millisec = annotation["start_task_abs_ts_ms"] - positive_offset_ts_ms
        start_millisec = int(start_millisec)
    # handle default settings without n_samples
    if n_samples == -1:
        n_samples = \
            (end_task_in_video_millisec - start_millisec) // every_n_millisec
    n_samples = int(n_samples)


    # read iTracker plugin data
    df_snap = read_textual_snapshot(user_folder)
    earliest_itracker_data_ts_ms  = df_snap["time"].min()
    print("Earliest Eye-Tracker Data (millisec): ", earliest_itracker_data_ts_ms)

    # read the eye tracker data
    df_eye = read_eye_tracker_fixations(
        user_folder=user_folder,
        task_nr=task_nr,
    )
    df_eye = offset_time_based_on_time_column(df_eye)

    # check if the eye tracker has enough data
    if earliest_itracker_data_ts_ms > start_millisec + positive_offset_ts_ms:
        raise Exception(
            f"The eye tracker data is not available before the participants"
            "opens a file. Try increasing the start_second value. "
        )

    all_attended_tokens = []
    n_fixation_outside_code_area = 0
    tmp_dir = tempfile.TemporaryDirectory(dir="./tmp")
    code_screen_coordinate_abs = None
    files_in_the_ide = []

    for i in tqdm(range(start_millisec, start_millisec + every_n_millisec * n_samples, every_n_millisec)):
        # end of task detection
        if i > end_task_in_video_millisec:
            print(f"End of task according to the annotation. {annotation['task_end']}")
            break

        if debug:
            fig, axes = plt.subplots(1, 3, figsize=(30, 5.5))
            ax_video = axes[0]
            ax_itracker = axes[1]
            ax_eye = axes[2]

        # get video frame
        video_time_millisec = i
        video_time_frame = round(video_time_millisec / 1000 * fps)
        # we need the screenshot when we do not have the coordinates of the
        # code area
        if (not assume_code_area_is_not_resized or \
                not code_screen_coordinate_abs):
            img, img_path = get_video_image(
                video_path=video_path,
                frame_number=video_time_frame,
                output_folder=tmp_dir.name)
        if debug:
            ax_video.imshow(img)
            ax_video.set_title(f"Video (frame: {video_time_frame}, sec: {video_time_millisec/1000})")

        # get the itracker data
        itracker_time_millisec = video_time_millisec + positive_offset_ts_ms + int(annotation["shift_text_ms"])
        record = get_textual_snapshot_at(df_snap, time_ms=itracker_time_millisec)

        # keep track of which file is displayed on the screen
        # because the most frequently display file will be the task file
        file_path = record["source_filename"]
        file_name = re.search(
            "([a-zA-Z0-9 _.]+):[0-9]+\n$", file_path).group(1)
        folder = re.search(
            "\\\\([a-zA-Z]+)\\\\[a-zA-Z0-9 _.]+:[0-9]+\n$", file_path).group(1)
        files_in_the_ide.append(join(folder, file_name))

        snapshot_line_start = int(
            re.search(":([0-9]+)\n$", record["source_filename"]).group(1))
        if debug:
            display_text(
                content="".join(record["content_lines"]).replace("\t", "    "),
                font_size=10,
                show_line_numbers=True,
                debug=False,
                ax=ax_itracker,
            )
            ax_itracker.set_title(f"iTracker ({itracker_time_millisec} ms)")

        # get the eye-tracker data
        eye_tracker_fixation_time_millisec = \
            video_time_millisec + positive_offset_ts_ms + int(annotation["shift_eye_data_ms"])
        current_fixation = get_eye_tracker_fixations_at(df_eye,
            time_ms=eye_tracker_fixation_time_millisec)
        fixation_X = current_fixation["FPOGX"]
        fixation_Y = current_fixation["FPOGY"]

        if user_zoom == "M":
            marker_params = {
                "path_marker_top_left": "../markers/top_left_marker.png",
                "path_marker_bottom_right": "../markers/bottom_right_marker.png",
                "top_left_offsets": (182, 109),
                "bottom_right_offsets": (107, 6),
            }
            col_lines_params = {"n_lines": 26, "n_col": 97,}
        elif user_zoom == "L":
            marker_params = {
                "path_marker_top_left": "../markers/top_left_marker_zoomed.png",
                "path_marker_bottom_right": "../markers/bottom_right_marker_zoomed.png",
                "top_left_offsets": (220, 125),
                "bottom_right_offsets": (90, -25),
            }
            col_lines_params = {"n_lines": 20, "n_col": 71,}

        # attribute fixation points to characters

        if (not assume_code_area_is_not_resized or \
                not code_screen_coordinate_abs):
            code_screen_coordinate_abs = get_code_coordinates(
                screenshot_path=img_path,
                percentage_format=False,
                debug=False,
                **marker_params)

        optional_debug_args = {}
        if debug:
            ax_eye.set_title(f"Eye-Tracker ({eye_tracker_fixation_time_millisec} ms)")
            optional_debug_args = {
                "ax": ax_eye,
                "screen_path": img_path}
        (main_char_line_index, main_char_col_index) = get_char_coordinates(
            eye_x_perc=fixation_X, eye_y_perc=fixation_Y,
            pixel_screen_width=1920, pixel_screen_height=1080,
            debug=debug,
            **optional_debug_args,
            **col_lines_params,
            **code_screen_coordinate_abs)

        if main_char_line_index == 0 or main_char_col_index == 0:
            if debug:
                print("Looking outside of the screen. No attention to consider.")
            n_fixation_outside_code_area += 1
            continue

        attended_tokens = convert_char_coordinate_to_attention_area(
            line=main_char_line_index,
            col=main_char_col_index,
            content="".join(record["content_lines"]).replace("\t", "    "),
            content_starting_line=snapshot_line_start,
            horizontal_attention_span=horizontal_attention_span,
            vertical_attention_span=vertical_attention_span,
            tot_screen_col=col_lines_params["n_col"],
            tot_screen_lines=col_lines_params["n_lines"],
            debug=debug
        )
        all_attended_tokens.extend(attended_tokens)


    tmp_dir.cleanup()
    print(
        f"There were {n_fixation_outside_code_area} " +
        "fixations outside the code area."
    )
    # get the most frequent string shown in the IDE, that is the task
    target_file_path = Counter(files_in_the_ide).most_common(1)[0][0]
    return target_file_path, all_attended_tokens


def add_character_based_on_source_file(
        df: pd.DataFrame,
        source_file_path: str,
        tot_screen_lines: int = 26,
        tot_screen_col: int = 97,):
    """Add a token column based on the `line` and `column` columns.

    The token represents the character at the `line` and `column` position
    in the input source file.
    The input source code file will be padded to reach the desired line length
    (i.e. by adding spaces) and file length (i.e. by adding lines).
    """
    with open(source_file_path, "r") as f:
        source_code = f.read()
    source_code_lines = source_code.split("\n")
    padded_lines = []
    for i, line_content in enumerate(source_code_lines):
        # print(f"{i} {line_content}")
        i_padded_line = line_content + "\n" + " " * tot_screen_col * 4
        padded_lines.append(i_padded_line)
    # add empty lines at the end of the file
    padded_lines.extend(
        [" " * tot_screen_col * 4 for _ in range(tot_screen_lines * 2)])
    # from pprint import pprint
    # pprint(padded_lines)
    # print(f"Smallest col: {df['column'].min()}")
    # print(f"Largest col: {df['column'].max()}")
    # print(f"Smallest line: {df['line'].min()}")
    # print(f"Largest line: {df['line'].max()}")
    # print(f"Total lines: {len(padded_lines)}")
    df["token"] = df.apply(
        lambda row: padded_lines[int(row["line"]) - 1][int(row["column"]) - 1],
        axis=1
    )
    return df


def convert_char_occurrences_to_tokens_and_weights(
        char_level_coordinates: List[Tuple[int, int]],
        source_code_file_path: str,
    ):
    """Convert char coordinates to codeattention tokens and weights.

    It will output the  char-level tokenization used by codeattention library,
    and the corresponding attention weights.

    Example output:
    - tokenization: List[Dict[str, Any]]
        e.g.
        [{'i': 0, 'l': 0, 'c': 1, 'w': 4, 't': ' '},
        {'i': 1, 'l': 0, 'c': 2, 'w': 9, 't': ' '},
        {'i': 2, 'l': 0, 'c': 3, 'w': 9, 't': ' '},
        {'i': 3, 'l': 0, 'c': 4, 'w': 9, 't': ' '},
        {'i': 4, 'l': 0, 'c': 5, 'w': 9, 't': ' '}]
    - attention_weights: List[float]
        the ordered list of weights (the w field):
        e.g.
        [4, 9, 9, 9, 9]
    """
    char_records = [
        {"line": c[0], "column": c[1]} for c in char_level_coordinates]
    df = pd.DataFrame.from_records(char_records)
    df = df.groupby(by=["line", "column"]).size().reset_index(name="count")
    df = df.sort_values(by="count", ascending=False)

    df_w_token = add_character_based_on_source_file(
        df, source_file_path=source_code_file_path, tot_screen_lines=26, tot_screen_col=97)
    df_w_token.head()
    df_w_token_sorted = df_w_token.sort_values(by=["line", "column"])
    formatted_tokens = df_w_token_sorted.rename(
        columns={"line": "l", "column": "c", "token": "t", "count": "w"}
    ).to_dict(orient="records")
    formatted_tokens = [
        {'i': i, **ft} for i, ft in enumerate(formatted_tokens)
    ]
    only_weights = [ft["w"] for ft in formatted_tokens]
    return formatted_tokens, only_weights


def convert_char_attribution_to_tokens_and_weights(
        df_char_attribution: pd.DataFrame,
        source_code_file_path: str,
        tot_screen_lines: int,
        tot_screen_col: int,
    ):
    """Convert char coordinates to codeattention tokens and weights.

    The input is a list of coordinates with the time elapsed on them:
    - line: int
    - col: int
    - time_spent: float
    There might be duplicates
    It will output the  char-level tokenization used by codeattention library,
    and the corresponding attention weights.
    Note that the input df is expected to have columns and line starting at 1.
    The output will follow the same convention

    Example output:
    - tokenization: List[Dict[str, Any]]
        e.g.
        [{'i': 0, 'l': 1, 'c': 1, 'w': 4, 't': '/'},
        {'i': 1, 'l': 1, 'c': 2, 'w': 9, 't': '*'},
        {'i': 2, 'l': 1, 'c': 3, 'w': 9, 't': '*'},
        {'i': 3, 'l': 1, 'c': 4, 'w': 9, 't': '*'},
        {'i': 4, 'l': 1, 'c': 5, 'w': 9, 't': '*'}]
    - attention_weights: List[float]
        the ordered list of weights (the w field):
        e.g.
        [4, 9, 9, 9, 9]
    """
    df = df_char_attribution
    df = df.groupby(by=["line", "column"]).sum().reset_index()

    df_w_token = add_character_based_on_source_file(
        df, source_file_path=source_code_file_path,
        tot_screen_lines=tot_screen_lines, tot_screen_col=tot_screen_col)
    df_w_token_sorted = df_w_token.sort_values(by=["line", "column"])
    formatted_tokens = df_w_token_sorted.rename(
        columns={"line": "l", "column": "c", "token": "t", "time_spent": "w"}
    ).to_dict(orient="records")
    formatted_tokens = [
        {'i': i, **ft} for i, ft in enumerate(formatted_tokens)
    ]
    # res = [print(t["t"], end='') for t in formatted_tokens]
    only_weights = [ft["w"] for ft in formatted_tokens]
    formatted_tokens = [t for t in formatted_tokens if t['t'] is not None]
    assert len(formatted_tokens) == len(only_weights), \
        "The number of tokens and weights is not the same."
    return formatted_tokens, only_weights