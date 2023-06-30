
# Extracting Meaningful Attention on Source Code: An Empirical Study of Developer and Neural Model Code Exploration

This project showcase some possible uses of the attention signal of neural model of code, in particular we focus on generative GPT-like model, thus we extract the attention from the transformer units.
Note that GPT-like models are decoder only, thus we get masked attention (i.e. only the previous tokens get attended) and the attention matrix is triangular.

We collect ground truth eye tracking data from developers while they explore code, and we compare the attention signal of the neural model with this ground truth data.

## Sense-Making Task

We study the models and developers when performing the sense-making task: given a self-contained source code file, the task is answer a single question (e.g. complexity related, parameter related, etc.) listed at the end of the file. Starting from the prompt: `# Answer:`.

## Data

- `post-processed eye tracking data and ready-to-plot experimental data`: contains the post processed data from the raw data, including the ground truth visual attention vectors, the interaction matrix, and the anntation on the answer correctness. Available [here](https://figshare.com/s/c11c1ad03dcf4e0126c5)
- `raw eye tracking data`: contains the raw data from 25 developers over 92 valid code exploration sessions divided in two batches. Available [here](https://figshare.com/s/08a8b67349f18007376e). Place the content at the path: `data/eye_tracking_studies/dataset_human_sessions`.
- `sensemaking code snippets`: list of files used for the eye tracking study and to be processed by the neural models. Available [here in the repo](data/prompts_collection/sensemaking).


## Installation

Follow these steps to prepare the repository to reproduce the plots in the paper:
1. setup a virtual environment in the current folder:
```bash
pip install virtualenv
virtualenv venv
```
2. activate the virtual environment:
```bash
source venv/bin/activate
```
3. install all the dependencies:
```bash
pip install -r config/dependencies/venv_requirements.txt
```

## Reproduce Plots from Exeperimental Data - Level 1

This level of reproduction allows to reproduce the plots in the paper, without running the experiments from scratch.

1. Download the `experimental_data` and place the content in the `data/experimental_data` folder.
The data included in the `experimental_data` folder are:
    - The folder `cmp_v06` contains the comparison csv generated for all the models, check the config files in `config/comparisons/cmp_v06*` for the exact configuration used for the different models.
    - The folder `eye_v10` is contains the metadata regarding how much time has been spent on each token (using tokens of different models). For more details check the config files in `config/eye_v10*` for the exact configuration used for the different models.

Note to reproduce these analysis from scratch use the readme and the `.sh` scripts in the main folder of the repository.

2. Run the notebook `notebooks/47_Empirical_Study_Results.ipynb` to reproduce the plots in the paper.


## Reproduce With New Large Language Models - Level 2

This level of reproduction allows to recreate all the experimental data, from the attention extraction to the comparison with the data in the eye tracking dataset.

**Preliminaries**:
- Note: you need the `screen` command to run the experiments, you can get via `sudo apt-get install screen`.
- Download the human data from [here](https://figshare.com/s/08a8b67349f18007376e), unzip and place the content at the path: `data/eye_tracking_studies/dataset_human_sessions`.


1. Run the script `1_DOWNLOAD_MODELS.sh` to download the models used in the experiments. Insert the model name you want to study and its HuggingFace identifeir (e.g., `Salesforce/codegen-350M-mono`). Note that this piepline works only with HuggingFace models. Then insert a folder where you want to download your model locally (e.g. `data\models`). The models will be downloaded in the `data/models` folder. This step will generate a config file with the details of your experiment in the folder `config/automatic`, with the name `download_{timestamp}.yaml`.

2. Run the script `2_QUERY_MODEL.sh` to query the model with the code snippets form the sensemaking task followed by the questions. First you have to decide which configuration to use among those in the folder `config/template/experiments`. For demo purposes, we suggest `exp_vXX_test.yaml`, whereas to be consistent with the paper use the `exp_v10_rebuttal.yaml`.
When prompted you have to choose a short name for your model (e.g. `codegen350mono`), then the output will be stored here: `data/model_output/exp_vXX_test/codegen350mono`.
This will generate a config file with the details of your experiment in the folder `config/automatic`, with the name `{template_name}_{modelname}_{date}.yaml` and output the attention signal of the model and its derived metrics in the folder: `data/model_output/exp_vXX_test/codegen350mono`.

3. Run the script `3_COMPARE_WITH_HUMAN.sh` to compare the attention signal of the model with the ground truth data. First you have to decide which configuration to use among those in the folder `config/template/comparisons`.
For demo purposes, we suggest `cmp_vXX_test.yaml`, whereas to be consistent with the paper use the `cmp_v10_rebuttal.yaml`.
When prompted you have to choose a short name for your model (e.g. `codegen350mono`) use the same name as done in the previous step.
The it will ask which configuration to use to postporocess the eye tracking data form the humans. For demo purposes, we suggest `eye_vXX_test.yaml`, whereas to be consistent with the paper use the `eye_v10_rebuttal.yaml`.
This will generate a config file with the details of your experiment in the folder `config/automatic`, with the name `{template_name}_{modelname}_{date}.yaml` and output the comparisons in the folders: `data/eye_tracking_attention/eye_vXX` and `data/comparisons/cmp_vXX`.


## Table of Content

This repository contains the following subfolders:
- `attwizard`: all the code and scripts ([`attwizard.script`]) used to manipulate, analyze and visualize the attention signal.
Note that this packge includes also tools to post-process data from the HRR and for comparing data from human and models.
- `eye_tracking`: the code and scripts used to post-process eye tracking data collected during the eye tracking study.
- `config`: the configuration files used in the experiments.
- `data`: the data collected in the experiments.
- `notebooks`: the notebooks used to design and prototype the experiments.

## Advanced
- (optional) if you want to store your experiment in Azure container storage, you need to setup a blobfuse and mount the container. Then you will rely on a `fuse_connection.cfg` file to store in the root of the repo. It will contain the following
```bash
accountName yourStorageAccountName
accountKey yourStorageAccountKey
containerName yourContainerName
```
To link the local folder to an Azure container storage, you can use the following command:
```bash
PREPARE_CONTAINER.sh
```
Then follow the instruction in the terminal.
- Check the [attwizard/script](attwizard/script) folder for the scripts used to manipulate, analyze and visualize the attention signal.

