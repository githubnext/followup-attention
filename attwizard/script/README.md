Here we describe the scripts available in this folder and how to call them.
Remember to call them from the main repository folder with the following command
```bash
python -m attwizard.script.scrip_name ...
```

# Download the Models (locally)

To download the models you can use the following command:
```bash
python -m attwizard.script.download_model
```
Bear in mind that the model are listed in the `models` key of the [`config/download_config.yaml`](config/download_config.yaml) file.


# Download and Process Human Data

The `human_data.py` has the following functions:
```
Usage: python -m attwizard.script.human_data [OPTIONS] COMMAND [ARGS]...

Options:
  --config TEXT  Pass path to yaml config file with keys: mongo_db_url,
                 raw_data_folder, att_weights_folder, att_heatmap_folder,
                 task_metadata_folder.
  --help         Show this message and exit.

Commands:
  download     Download the human data.
  getheatmaps  Create the heatmaps for all the nicknames in the database.
  getweights   Derive the attention weights for each submission.
```
An example to download the data would be:
```bash
python -m attwizard.script.human_data --config config/hum_vXX.yaml download
```
Change the last keyword to perform other commands.


# Create Custom Prompts


To create prompt in a structure way, we have a base class `prompt_creator` class, available in the `attwizard/script/prompt_creator.py` file. Future classes should be derived from that class, one example is the class `AssignmentChainCreator` (in this [file](attwizard/creators/assignement_chain.py)).
Remember to refer it in the config file that you are using a `creator` type: `config["prompt_creation_strategy"]["type"] = "creator"`

To use it call the following command:

```bash
python -m attwizard.script.experiment_creation --config config/exp_vXX.yaml creator
```

The legacy create function is still available for some cases (e.g. gibberish prompts). you can use the following command:
```bash
Usage: python -m attwizard.script.experiment_creation [OPTIONS] COMMAND
                                                      [ARGS]...

Options:
  --config TEXT  Pass path to yaml config file with keys:
                 prompt_creation_strategy, input_data_folder.
  --help         Show this message and exit.

Commands:
  create  Create new prompts according to different strategies.
```
An example would look like:
```bash
python -m attwizard.script.experiment_creation --config config/exp_vXX.yaml create
```
Refer to exp_v03.yaml for an example config of the legacy method.


# Query Models and Store Completions, Attention and Metadata

You have to run the code to extract the attention weights from the snippets in the `input_data_folder` folder (as mentioned in the config file):
```bash
Usage: python -m attwizard.script.batch_attention_extraction
           [OPTIONS] COMMAND [ARGS]...

Options:
  --config TEXT  Pass path to yaml config file with keys: input_data_folder,
                 output_data_folder, local_model_folder, stop_signal_line,
                 generation, attention, models.
  --help         Show this message and exit.

Commands:
  queryextract  Query the model for completion and extract attention.
```
An example of usage to query the model and get attention weights would be:
```bash

python -m attwizard.script.batch_attention_extraction --config config/exp_vXX.yaml queryextract
```
