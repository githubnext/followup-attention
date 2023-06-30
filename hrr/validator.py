
import click
import yaml
from pymongo import MongoClient
import pandas as pd
import os
import pathlib
import json
from typing import List, Tuple, Union, Any, Dict
from datetime import datetime
import shutil

from hrr.validation.google_form_data import download_form_submissions
from hrr.validation.amturk_data import download_work_submissions
from hrr.validation.amturk_data import get_nickname_to_review
from hrr.validation.amturk_data import  get_nickname_in_status
from hrr.validation.main_task_data import txt_report_nickname
from hrr.validation.main_task_data import get_tasks_nickname_with_score
from hrr.post_processing import getweights

from hrr.utils import save_with_timestamp
from hrr.utils import save_if_not_existing
from hrr.utils import get_most_recent



from attwizard.script.utils_mongodb import load_all_aggregated_records


@click.group()
@click.option(
    '--config', default=None,
    help="Pass path to yaml config file with keys: " +
         "path_mongo_db_url, " +
         "evaluation_folder, " +
         "spreadsheet_id, " +
         "spreadsheet_range.")
@click.pass_context
def cli(ctx, config):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    # read yaml config file
    ctx.obj['CONFIG_FILEPATH'] = config
    if config is not None:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        ctx.obj['CONFIG'] = config


@cli.command()
@click.option(
    '--path_mongo_db_url', default=None,
    help='Path to a file containing a Mongo url string: mongodb://root:password@mongodb.com:27017/')
@click.option(
    '--evaluation_folder', default=None,
    help='Output folder where to store evaluation data.')
@click.pass_context
def download(ctx, path_mongo_db_url, evaluation_folder):
    """Download the google form, AMTurk data and main tasks."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)
    if config:
        click.echo("Using config file information.")
        path_mongo_db_url = config.get('path_mongo_db_url', None)
        mongo_db_url = open(path_mongo_db_url, 'r').read().strip()
        evaluation_folder = config.get('evaluation_folder', None)
        n_hours_min_age = config.get('n_hours_min_age', None)
    amturk_dir = os.path.join(evaluation_folder, 'amturk')
    google_form_dir = os.path.join(evaluation_folder, 'google_form')
    main_task_dir = os.path.join(evaluation_folder, 'main_task')
    summmary_dir = os.path.join(evaluation_folder, 'summary')

    dirs = [amturk_dir, google_form_dir, main_task_dir, summmary_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    df_amturk = download_work_submissions(
        path_to_credentials=config["path_amturk_credentials"],
        output_folder=amturk_dir,
        target_hit_id=config["target_hit_id"]
    )

    df_form = download_form_submissions(
        path_to_credentials=config["path_google_credentials"],
        spreadsheet_id=config.get('spreadsheet_id', None),
        spreadsheet_range=config.get('spreadsheet_range', None),
        output_folder=google_form_dir)

    df_merged = pd.merge(df_amturk, df_form, on='WorkerId', how='left')
    save_with_timestamp(
        data=df_merged, folder=summmary_dir,
        prefix="summary", extension=".csv")

    mongo_client = MongoClient(mongo_db_url)
    all_data = load_all_aggregated_records(
        mongo_client,
        keep_only_submitted=False,
        min_age_in_seconds=60*60*n_hours_min_age,
        remove_warmup=True)
    mongo_client.close()
    save_if_not_existing(
        data=all_data, output_folder=main_task_dir)

    # Display results
    print("Workers' Scores:")
    for i, row in df_merged.iterrows():
        print(f"({row['AssignmentStatus']})", " workerId: ", row['WorkerId'], " - perc_correct: ", row['perc_correct'])


@cli.command()
@click.option(
    '--submitted', default=False, is_flag=True)
@click.option(
    '--approved', default=False, is_flag=True)
@click.option(
    '--create_reports', default=False, is_flag=True)
@click.option(
    '--create_summary_csv', default=True, is_flag=True)
@click.pass_context
def scoreusers(ctx, submitted, approved, create_reports, create_summary_csv):
    """Score the performance of each user.

    This command scores the performance of each user.
    If the flag --submitted is passed, only submitted users are scored, namely
    those who submitted work on AMTurk.
    If the flag --create_reports is passed, a report is created for each of
    those participants.
    """
    config = ctx.obj.get('CONFIG', None)
    click.echo("Using config file information.")
    evaluation_folder = config.get('evaluation_folder', None)
    google_form_dir = os.path.join(evaluation_folder, 'google_form')
    main_task_dir = os.path.join(evaluation_folder, 'main_task')
    amturk_dir = os.path.join(evaluation_folder, 'amturk')
    score_dir = os.path.join(evaluation_folder, 'score')
    user_summary_dir = os.path.join(evaluation_folder, 'user_summary')
    if create_reports:
        pathlib.Path(score_dir).mkdir(parents=True, exist_ok=True)
    if create_summary_csv:
        pathlib.Path(user_summary_dir).mkdir(parents=True, exist_ok=True)

    all_file_names = os.listdir(main_task_dir)
    if (submitted):
        # filter to only submitted ones
        # for which we need to take a payment decision
        submitted_nicknames = get_nickname_to_review(amturk_dir)
        nicknames_to_score = submitted_nicknames
        print("Scores shown for:")
        print(nicknames_to_score)
    elif (approved):
        approved_nicknames = get_nickname_in_status(amturk_dir, 'Approved')
        nicknames_to_score = approved_nicknames
        print("Scores shown for:")
        print(nicknames_to_score)
    else:
        # get all the nicknames
        all_nicknames = list(set([
            file_name.split('_')[0] for file_name in all_file_names]))
        nicknames_to_score = all_nicknames
    most_recent_google_form = get_most_recent(
        folder=google_form_dir,
        prefix="google_form_data", extension=".csv")
    df_google_form = pd.read_csv(
        os.path.join(google_form_dir, most_recent_google_form))
    for nickname in nicknames_to_score:
        c_nick_files = [
            fn for fn in all_file_names if fn.startswith(nickname + "_")]
        record_for_this_nickname = []
        for filename in c_nick_files:
            with open(os.path.join(main_task_dir, filename), 'r') as f:
                record_for_this_nickname.append(json.load(f))
        out_filename = None
        if create_reports:
            out_filename = os.path.join(score_dir, f"{nickname}.txt")
        txt_report_nickname(
            nickname, record_for_this_nickname,
            df_google_form=df_google_form,
            output_filepath=out_filename)
        if create_summary_csv:
            user_tasks = get_tasks_nickname_with_score(
                nickname, record_for_this_nickname, df_google_form,
            )
            df_user = pd.DataFrame.from_records(user_tasks)
            df_user.to_csv(
                os.path.join(user_summary_dir, f"{nickname}.csv"),
                index=False
            )

def evaluate_user(df_user: pd.DataFrame) -> Tuple[bool, str]:
    """Evaluate a participant and take an approval/rejection decision."""
    survey_score = df_user["survey_score"].iloc[0]
    average_time_per_task = df_user["task_duration_seconds"].mean()
    min_time_per_task = df_user["task_duration_seconds"].min()
    answers = list(df_user["answer"])
    all_answered = all([answer.strip() != "" for answer in answers])
    question_and_answers = list(zip(list(df_user["question"]), list(df_user["raw_answer"])))
    question_and_answers_str = "\n".join([f"{q} - {a}" for q, a in question_and_answers])
    main_tasks = [f.split("_")[0] for f in list(df_user["sourceFile"])]
    same_main_task_shown_twice = len(main_tasks) != len(set(main_tasks))
    n_sessions = len(df_user["randomcode"].unique())
    df_user["openedPageDatetime"] = df_user["openedPageTime"].apply(
        lambda x: datetime.fromtimestamp(x/1000))
    records = df_user[["randomcode", "openedPageDatetime", "completejsexecution", "completed", "task_duration_seconds"]].to_dict(orient="records")
    decision = {}

    if survey_score < 7:
        decision["approved"] = False
        decision["message"] = "Your first submission to the warm-up task contained mistakes that showed how you did not have the sufficient knowledge to perform the task. You ignored the warning provided at the end of the warm-up task task and submitted your work even if you could have realized that you did not completed all the answers correctly."
        decision["reason"] = f"survey_score: {survey_score}"
        return decision


    if len(df_user) == 0:
        decision["approved"] = False
        decision["message"] = "You did not complete the main task."
        decision["reason"] = "n records = 0"
        return decision

    if same_main_task_shown_twice and n_sessions > 1:
        decision["approved"] = False
        decision["message"] = "Your workid appears at least twice in the main task, breaking the requirement that the submitted answer must be from a participant who has never seen those code snippets before."
        decision["reason"] = f"submissions: " + str(records)
        return decision

    if not all_answered:
        decision["approved"] = False
        decision["message"] = "Some questions of the main task (if not all) were not answered."
        decision["reason"] = f"answers: " + question_and_answers_str
        return decision

    if average_time_per_task < 120 or min_time_per_task < 30:
        decision["approved"] = False
        decision["message"] = "You spent too little time to be compatible with a credible code inspection."
        decision["reason"] = f"average_time_per_task: {average_time_per_task}, min_time_per_task: {min_time_per_task} : " + str(records)
        return decision

    decision["approved"] = True
    decision["message"] = "Approved. Good Job."
    decision["reason"] = f"average_time_per_task: {average_time_per_task}, min_time_per_task: {min_time_per_task} : " + str(records) + f"answers: {question_and_answers_str}"
    return decision


@cli.command()
@click.pass_context
def getapprovedattention(ctx):
    """Get the attention weights for the approved users.

    Note this relies on the most recent download of the amturk data.
    So make sure to download the latest amturk data with the download command
    of this script:
    python -m hrr.validator --config config/hum_vXX.yaml download
    """
    config = ctx.obj.get('CONFIG', None)
    config_filepath = ctx.obj.get('CONFIG_FILEPATH', None)
    click.echo("Using config file information.")
    click.echo("Extracting the attention weights form the approved sessions.")
    dir_accepted_sessions = config['accepted_data_folder']
    # create directo if it doesn't exist
    pathlib.Path(dir_accepted_sessions).mkdir(parents=True, exist_ok=True)
    dir_att_weights = os.path.join(
        config['derived_data_folder'], "att_weights")
    pathlib.Path(dir_accepted_sessions).mkdir(parents=True, exist_ok=True)
    evaluation_folder = config['evaluation_folder']
    amturk_dir = os.path.join(evaluation_folder, 'amturk')
    # identify the accepted users
    approved_nicknames = get_nickname_in_status(amturk_dir, 'Approved')
    nicknames_to_score = approved_nicknames
    print("Scores shown for:")
    print(nicknames_to_score)
    # save them to a txt files
    with open(os.path.join(dir_accepted_sessions, "approved_nicknames.txt"), 'w') as f:
        f.write("\n".join(approved_nicknames))
    # for each of the user (one line of the txt file)
    for nickname in approved_nicknames:
        # run the following command
        # python -m hrr.post_processing --config config/hum_vXX.yaml getweights --nickname USERNAME --with_heatmaps
        # where USERNAME is the nickname of the user
        os.system(f"python -m hrr.post_processing --config {config_filepath} getweights --nickname {nickname} --with_heatmaps")
        # then copy the newly generated files starting with the USERNAME to the dir_att_weights
        # get all the files starting with nickname in the dir_att_weights
        all_files = os.listdir(dir_att_weights)
        files_to_copy = [f for f in all_files if f.startswith(nickname)]
        for f in files_to_copy:
            shutil.copy(os.path.join(dir_att_weights, f), dir_accepted_sessions)


if __name__ == '__main__':
    cli()