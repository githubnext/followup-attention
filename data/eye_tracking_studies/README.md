
# Re-run the processing pipeline on our eye tracking data

To rerun the processing pipeline on our eye tracking data, follow these steps:
1. Download the raw data from the Figshare repository with this [link](https://figshare.com/s/d56af6f915bc29e6f620).
2. Unzip the data in the `data/eye_tracking_studies/dataset_human_sessions` folder.
3. Go to the relevant `eye_vXX` configuration file and update the reference to the batch data in the `batches` section. Make sure to have the following:
```yaml
batches:
  - batch_no: 1
    folder_questions: ./eye_tracking/eye_tracking_studies/dataset_human_sessions
    folder_participants: ./data/eye_tracking_studies/source_file_human_dataset_sessions
```
And make sure that all the `participants` are pointing to the batch number 1 just created.
4. Go to the eye tracking folder readme to see the next steps to run the pipeline.


# Conducting an eye tracking study from scratch
We assume that the responsible person for concucting the study has used the same eye tracker and the same VSCode plugin used in the study, and for each participant has created a folder in Google drive containing the eye fization data as `.csv` files and the VSCode plugin data in a `Snapshots` folder as `.txt` files.

## Data Download

Create a folder named `batch_XX` where `XX` is a number.
Then download your google drive data in that folder.

## Data Anonimization

Each batch goes through anonymization via a simple script:

```bash
# move the anonymization script to the folder with the participants sub-folder
cp anonymization.py batch_1/
# change the pointer to the right yaml file with the mapping
# on the line: anonymize(
#    anonymization_dict_file_path="anonymization_mapping.yaml")
# run the script for each participant
python anonymize.py
```

You need a yaml file like this:

```yaml
real_name_1: UrandomU
real_name_2: UsampleU
...
```
where `real_name_1` is the name of the participant in the original data and `UrandomU` is the name of the participant in the anonymized data.
This will override all folder, subfolder and file names with the anonymized names.