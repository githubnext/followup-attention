#!/bin/bash


# get current working directory
current_dir=$(pwd)

# read the name of the new experiment
echo "Enter the name of the experiment (e.g. eye_vXX or exp_vXX) to move to ramdisk:"
read -p "Experiment name to move: " experiment

# check if the experiment name contains eye
if [[ $experiment == *"eye"* ]]; then
    data_dir="data/eye_tracking_attention/$experiment"
fi

# check if the experiment name contains exp
if [[ $experiment == *"exp"* ]]; then
    data_dir="data/model_output/$experiment"
fi

# create a new ramdisk
echo "Creating new ramdisk of 20 GB..."
mkdir -p ramdisks/$experiment
echo "Creating new ramdisk...done"
mount -t tmpfs -o size=20G tmpfs $current_dir/ramdisks/$experiment

echo "Unmounting old blobfuse: $data_dir ..."
fusermount -u $current_dir/$data_dir
echo "Unmounting old blobfuse: $data_dir ...done"

# re-mount to the new ramdisk

# remove the underscore from the experiment name
experiment_escaped=${experiment//_/}
echo "Experiment name escaped: $experiment_escaped"

# remove the last line of the fuse_connection.cfg file
head -n -1 fuse_connection.cfg > temp.txt  ; mv temp.txt fuse_connection.cfg

# append the new Container name to the fuse_connection.cfg file
echo "containerName $experiment_escaped" >> fuse_connection.cfg

# connect the two containers
blobfuse $current_dir/$data_dir --tmp-path=$current_dir/ramdisks/$experiment \
 --config-file=fuse_connection.cfg \
 -o attr_timeout=240 \
 -o entry_timeout=240 \
 -o negative_timeout=120

echo "Connection established for $experiment ($data_dir --> $experiment_escaped (Azure Container))"
echo "(tmp folder in Ramdisk $current_dir/ramdisks/$experiment)"
