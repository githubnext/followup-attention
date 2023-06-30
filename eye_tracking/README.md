To run the attention extraction from the eye tracking data and the snapshot from the vscode plugin you can use the `eye_tracking_post_processing.py` tool.

# Check if the code area is correctly detected
Run this script to create images of the code screen and the grid with lines and columns of the code area.
Here you can verify that the grid (what the algorithm sees) matches code area in the screenshot (what the developer sees).
Run this command:
```bash
python -m eye_tracking.test_and_dry_runs config/eye_vXX.yaml inspectgrids
```

# Run Single user Single task Extraction

To extract for a specific user and task, run the following command, from the root directory:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml getweights --username UconsumerU --zoom M --task_number 1 --override
```
Note that the task number refers to the video which has to be inspected to extract the x and y coordinates, whereas the zoom refers to the zoom with which the user run the experiment, it can either be `M` or `L`, check the `eye_v01.yaml` file for the appropriate one.
Other parameters can be passed via the `eye_v01.yaml` file.

# Get Model-specific weights

The weights extracted from the eye tracker data are naturally at the char level. In case you want them at the level of the token that also the model sees you can convert them with the follwing command:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml derivetokenweights
```

Check the `eye_v10_gptj.yaml` file for an example in the `tokenization` field.
This will create a subfolder for each name in the field `model_names` in the `folder_output` folder.
Note that it reads and convert all the data from the `att_weights` folder, so make sure to run the `getweights` command before.

# Run all user and all tasks extraction

To run the extraction for all the user (mentioned in the `participants` field of the `eye_vXX.yaml` file) run the following command:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml getweights --override
```

# Derive the followup matrices: User interactions

To derive the followup matrices which tells how often a user was looking at token A after having looked at token B, you have to run the following command:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml getfollowup --override
```

Note that this produces also the line information, aka how ofter a user looked at line X after having looked at line Y.
These outputs are stored both as images and as numpy objects `.npy`.

## More info
For more info read the [`eye_v01.yaml`](../config/participants.yaml) file.
Each field is documented there.


# Get Baseline Dataset Behavior

Humans might have a baseline behavior, which is the behavior they would have if they were at the code irrespective of the content (e.g. they tend to read what comes next and they might look back at the previous line to understand what the context is).
To get the baseline behavior, run the following command:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml getavgbehavior
```

It relies on the `getfollowup` command, so make sure to run it before.
It will produce two `.csv` files, one with raw data for each new token movement
and one with the average behavior of the users when looking at n tokens away.
They are stored in the same `decay` folder of the respective followup data.


# Grading of the Answers
To grade the asnwers of the participants you can create a datarame containing
them. Run the following commands to create the `answer_dataset.csv` file in the folder deterimined by the `folder_output` field in the `eye_vXX.yaml` file:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml getanswers
python -m eye_tracking.post_processing config/eye_vXX.yaml getanswersdataset
```
