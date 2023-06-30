

# HRR Participant - Data Download

Run the following command to download all the relevant data for evaluation of the workers, namely:
- google form submissions
- amazon mechanical turk submissions and workerIds
- human reasoning recorder tracking data on the main task

```bash
python -m hrr.validator --config config/hum_vXX.yaml download
```

Check `hum_v03.yaml` for an example of configuration.

## HRR Participant - Data Inspection

To see some key statistics about the task of each worker run this command:

```bash
python -m hrr.validator --config config/hum_vXX.yaml scoreusers
```
Check `hum_v05.yaml` for an example of configuration.

To focus only on the submissions which are pending on AMTurk, add the flag `--submitted` at the end.
Use the `--create_reports` flag to create a report for each worker in the `score` subfolder.

## HRR Participant - Deriving Weights (and Code Heatmaps)

To derive the attention weights of a specific user run the following command:

```bash
python -m hrr.post_processing --config config/hum_vXX.yaml getweights --nickname AXXXXXXXXXXXZ
```

Optionally you can also add the flag `--with_heatmaps` to create the heatmaps for each task.
Check `hum_v05.yaml` for an example of configuration.


## Grade users in AMTurk

To grade the users in AMTurk,
1. run the following commands to download the AMturk data and score the users:
    ```bash
    python -m hrr.validator --config config/hum_vXX.yaml download
    python -m hrr.validator --config config/hum_vXX.yaml scoreusers --submitted --create_reports
    ```
2. go to amturk and download the csv with the participants to grade and place it in the folder `data/hrr/hum_vXX/results`
3. open the notebook `27_Grade_HRR.ipynb` and set the two parameters to the right `hum_vXX` and `amturk_just_downloaded.csv`
4. run the notebook
5. fetch the csv `data/hrr/hum_vXX/results/rejected_users_YYYYYYY.csv` with the highest `YYYYYYY` number and upload it to AMTurk.

## Create Dataset of Attention Weights (Approved AMTurk Submissions)

To derive the attention weights of all approved AMTurk submissions run the following command:

```bash
# to download all the user data
python -m hrr.validator --config config/hum_vXX.yaml download
# to dataset of attention weights in the `accepted_data_folder` directory
python -m hrr.validator --config config/hum_vXX.yaml getapprovedattention
```
Check `hum_v06.yaml` for an example of configuration.
