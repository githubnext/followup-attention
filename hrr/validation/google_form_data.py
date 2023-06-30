"""This file validates the Google form submission.

Install these libraries:
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""


import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

import re
import pandas as pd
import boto3
import time

from hrr.utils import save_with_timestamp

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']


def download_form_submissions(
        path_to_credentials: str,
        spreadsheet_id: str,
        spreadsheet_range: str,
        output_folder: str,
        keep_log: str = True):
    """Download the google submissions.

    If keep_log is True, the existing file is kept and a new one is created.
    The convention is that the suffix of the csv, is the timestamp (e.g.
    google_form_data_1660756137.csv).
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(path_to_credentials):
        creds = Credentials.from_authorized_user_file(path_to_credentials, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                path_to_credentials, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(path_to_credentials, 'w') as token:
            token.write(creds.to_json())

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=spreadsheet_id, range=spreadsheet_range).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        print('Name, Major:')
        # Create dataframe and save
        df = pd.DataFrame(
            values[1:], columns=values[0])
        for row in values[:10]:
            # Print columns A and E, which correspond to indices 0 and 4.
            print('%s, %s' % (row[0], row[4]))
        # parse the first datatime column format e.g. 8/2/2022 15:17:32
        df['ts'] = pd.to_datetime(
            df['Timestamp'], format='%m/%d/%Y %H:%M:%S')
        # parse number of correct answers
        df['n_correct'] = df['Score'].str.split('/').str[0].astype(int)
        df['n_total'] = df['Score'].str.split('/').str[1].astype(int)
        df['perc_correct'] = ((df['n_correct'] / df['n_total']) * 100).round(2)

        # rename third column to WorkerId
        df.rename(columns={df.columns[2]: 'WorkerId'}, inplace=True)
        df['WorkerId'] = df['WorkerId'].str.strip()
        df.sort_values(by=['ts'], inplace=True)
        # remove duplicates and keep the first
        df.drop_duplicates(subset=['WorkerId'], keep='first', inplace=True)

        prefix = "google_form_data"
        if keep_log:
            save_with_timestamp(df, output_folder, prefix)
        else:
            df.to_csv(os.path.join(
                output_folder, '{prefix}.csv'), index=False)
        return df
