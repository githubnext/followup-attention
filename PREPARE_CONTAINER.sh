#!/bin/bash

# read the name of the new experiment
echo "Enter the name of the experiment (e.g. eye_vXX or exp_vXX or cmp_vXX):"
echo "=============="
echo "1. It can be a new or existing experiment."
echo "=============="
echo "2. You need to have a fuse_connection.cfg file in the current folder."
echo "With the following two lines as content:"
echo "---"
echo "accountName yourAccountName"
echo "accountKey yourSecretAccountKey"
echo "---"
echo "Note that you need to update the accountName and accountKey with your own values (found in the Azure Portal)."
echo "=============="
echo "3. You need to have a storage container already created before this script is run."
echo "You can create a storage container in the Azure Portal."
echo "=============="
echo "4. The storage container should have the same name as the experiment but without underscores (e.g. eyevXX, expvXX, cmpvXX)."
echo "=============="
read -p "Experiment name: " experiment

# create the new experiment tmp directory
mkdir -p /mnt/tmp_dir_blobfuse/$experiment

# check if the experiment name contains eye
if [[ $experiment == *"eye"* ]]; then
    echo "Creating subfolder in data/eye_tracking_attention..."
    # if the experiment name contains eye
    data_dir="data/eye_tracking_attention/$experiment"
    mkdir -p data/eye_tracking_attention/$experiment
fi

# check if the experiment name contains exp
if [[ $experiment == *"exp"* ]]; then
    echo "Creating subfolder in data/model_output and data/prompt ..."
    # if the experiment name contains exp
    data_dir="data/model_output/$experiment"
    mkdir -p data/prompt/$experiment
fi


# check if the experiment name contains cmp
if [[ $experiment == *"cmp"* ]]; then
    echo "Creating subfolder in data/model_output and data/prompt ..."
    # if the experiment name contains exp
    data_dir="data/comparisons/$experiment"
    mkdir -p data/comparisons/$experiment
fi

# check if the experiment is for a submission to a conference, aka if it contains iclr2023data
if [[ $experiment == *"iclr2023data"* ]]; then
    echo "Creating subfolder in data/iclr2023data ..."
    # if the experiment name contains iclr2023data
    data_dir="data/$experiment/"
    mkdir -p data/$experiment
fi

# create the data directory
mkdir -p $data_dir

# remove the underscore from the experiment name
experiment_escaped=${experiment//_/}
echo "Experiment name escaped: $experiment_escaped"

# keep only the first two lines from the fuse_connection.cfg file
head -n 2 fuse_connection.cfg > temp.txt  ; mv temp.txt fuse_connection.cfg

# append the new Container name to the fuse_connection.cfg file
echo "containerName $experiment_escaped" >> fuse_connection.cfg

# get current working directory
current_dir=$(pwd)

# connect the two containers
blobfuse $current_dir/$data_dir --tmp-path=/mnt/tmp_dir_blobfuse/$experiment \
 --config-file=fuse_connection.cfg \
 -o attr_timeout=240 \
 -o entry_timeout=240 \
 -o negative_timeout=120

echo "Connection established for $experiment ($data_dir --> $experiment_escaped (Azure Container))"
