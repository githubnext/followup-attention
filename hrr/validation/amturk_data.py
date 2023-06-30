"""Communicate with amturk endpoint."""
import pandas as pd
import boto3
import re
import os
from typing import Dict, List, Tuple, Union, Any

from hrr.utils import save_with_timestamp
from hrr.utils import get_most_recent


def download_work_submissions(
        path_to_credentials: str,
        output_folder: str,
        target_hit_id: str = None,
        keep_log: bool = True):
    """Download the amturk submissions.

    If no HIT id is passed the most recent one is used.
    """
    df_credentials = pd.read_csv(path_to_credentials)
    aws_access_key_id = df_credentials.iloc[0]['Access key ID']
    aws_secret_access_key = df_credentials.iloc[0]['Secret access key']
    MTURK_REAL = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client('mturk',
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        region_name='us-east-1',
        endpoint_url = MTURK_REAL #MTURK_SANDBOX #
    )
    print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")

    # get the HITs
    response = mturk.list_hits()
    hits = response['HITs']
    print("Active Hits:")
    hits = [
        {'id': h["HITId"], 'creation_time': h["CreationTime"]}
        for h in hits]
    # sort the hits by creation time
    hits = sorted(hits, key=lambda h: h['creation_time'])
    print(hits)
    # get the most recent HIT
    if target_hit_id is None:
        target_hit_id = hits[-1]['id']
    print("Downloading HIT: " + target_hit_id)

    # get all the assignments for the HIT via pagination
    assignments = []
    next_token_dict = {}
    while True:
        response = mturk.list_assignments_for_hit(
            HITId=target_hit_id,
            **next_token_dict
        )
        assignments.extend(response['Assignments'])
        next_token = response.get('NextToken')
        next_token_dict['NextToken'] = next_token
        if next_token is None:
            break


    df_amturk_api_submissions = pd.DataFrame(assignments)
    df_amturk_api_submissions['Answer'] = \
        df_amturk_api_submissions['Answer'].apply(
            lambda text:
                re.search('<FreeText>(.+?)</FreeText>', text).group(1))
    df_amturk_api_submissions.sort_values(by=['AssignmentStatus'], inplace=True)

    prefix = "amturk_submissions"
    if keep_log:
        save_with_timestamp(df_amturk_api_submissions, output_folder, prefix)
    else:
        df_amturk_api_submissions.to_csv(os.path.join(
            output_folder, '{prefix}.csv'), index=False)
    return df_amturk_api_submissions


def get_nickname_in_status(
        amturk_folder: str,
        status: str = "Submitted") -> List[str]:
    """Return the list of nicknames to review (check local data)."""
    most_recent_filename = get_most_recent(
        folder=amturk_folder,
        prefix="amturk_submissions",
        extension=".csv")
    df = pd.read_csv(os.path.join(
        amturk_folder, most_recent_filename))
    df = df[df["AssignmentStatus"] == status]
    return df['WorkerId'].values.tolist()


def get_nickname_to_review(amturk_folder: str) -> List[str]:
    """Return the list of nicknames to review (check local data)."""
    return get_nickname_in_status(amturk_folder, "Submitted")