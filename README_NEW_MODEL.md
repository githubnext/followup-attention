
# Evaluation of Other Models
To evaluate another model on this dataset follow these instructions.
Note that the pipelines works with huggingface models.
To run other models you have to aadapt the code yourself to query the model.

## Download the Model
You need to download the model using the file `config/download_config.yaml`, select the model by inserting huggingface address of the model in the `models_to_download` field as an element of the list.
Then run the following command from the root folder:
```bash
python -m attwizard.script.download_model
```
Remember that the model will be stored in the folder decided by the field `local_model_folder` in the `config/download_config.yaml` file, you will need to propagate this address also in the next config files.

## Run the Model

The steps are the following:
1. create a new configuration file in the `config` folder. Take inspiration from the existing ones, such as `exp_v10.yaml`.
Add your new model to the `models` section, by using its repository name on huggingface `author/model_name`.
2. install and source the virtual environment as explined in the main README.md
3. run the following command, from the main root of the repo:
```bash
screen -L -Logfile data/model_output/exp_v10/log_incoder.txt  python -m attwizard.script.batch_attention_extraction --config config/exp_v10_incoder.yaml queryextract
```
where `data/model_output/exp_v10/log_incoder.txt` points to a filepath which will be used to store the logs of the script.
Note that the folder of the log file must exist already.
Werease `config/exp_v10_incoder.yaml` is the path to the configuration file you created in step 1.

## Derive Attention Maps or Interaction Matrix from Model Data

Uncomment the analysis sections of the configuration file `exp_vXX` of your chouce, uncomment only those that you want to rerun, and run the following command:
```bash
screen -L -Logfile data/model_output/exp_v10/log_incoder.txt python -m attwizard.analysis_pass.analyzer --config config/exp_v10_incoder.yaml deriveall
```
To compare the vectors of attention uncomment the items: `naive_max`, `naive_mean`, `vector_naive_max_plus_mean_of_followers`, `vector_naive_mean_plus_mean_of_followers`.
Note that the items are computed sequentially, thus the later one can use the results of the previous ones.


## Process the Human Data
Follow the instruction in the `eye_tracking` folder to process the human data.
In particular the [Readme](eye_tracking/README.md) contains the instructions to process the data.
For example, you might want to compute the human attention on the specifc tokens used by the given model.

# Comparison: Human vs Model

1. create a new config file in the `config/comparisons` folder. Take inspiration from the existing ones, such as `cmp_v06.yaml`.
Remember to set the fields: `human_att_folder` to the folder with human processed data, `machine_att_folder` to the output folder of the configuration file of the previous step (field: `analysis_data_folder`), `output_comparison_folder` to a new folder where to store your output comparison (as `.csv` files).
2. run the following command to compare human vs model:
```bash
screen -L -Logfile data/comparisons/cmp_v06/log_incoder_rebuttal.txt python -m attwizard.analysis_pass.comparator --config config/comparisons/cmp_v06_incoder_rebuttal.yaml compare
```

## Comparisons with Interaction Matrix

Before running comparisons with the interaction matrix, you need to compute the base distribution used to normalize the calculation.
Humans might have a baseline behavior, which is the behavior they would have if they were at the code irrespective of the content (e.g. they tend to read what comes next and they might look back at the previous line to understand what the context is).
To get the baseline behavior, run the following command:

```bash
python -m eye_tracking.post_processing config/eye_vXX.yaml getavgbehavior
```

It relies on the `getfollowup` command, so make sure to run it before (see `eye_tracking/README.md` for more info).
It will produce two `.csv` files, one with raw data for each new token movement
and one with the average behavior of the users when looking at n tokens away.
They are stored in the same `decay` folder of the respective followup data.


# Compute Accuracy of the Model
To score a model you first have to extract the answers from the generated text and for this you can use the two following commands:
```bash
python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml getanswers
python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml getanswersdataset
```
This will create a `.csv` file with the machine answers in the `output_data_folder` specified in the configuration file.
it will focus on the first model specified in the `models` key of the configuration file.
For a full example you can have a look at `exp_v10.yaml`.
We suggest to open the csv with Excel or Google doc and manually annotate, then save the result as a new csv file with columns:
`filename`, `model_name`, `n_try`, `score`.
Note that score is supposed to be a number: 0, 1 or 2, where 2 is the best and 0 the minimum.
You can check the csv in the folder `data/annotations` for some example.
