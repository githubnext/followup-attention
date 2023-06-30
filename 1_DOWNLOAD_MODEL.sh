#!/bin/bash

# Query for the name of the huggingface repository
echo "Enter the name of the huggingface repository (e.g. facebook/incoder-6B):"
read -p "Huggingface repository name: " repository

# Ask to specify where the model should be saved, otherwise the default
# location is the directory: /mnt/huggingface_models
echo "Enter the path where the model should be saved (e.g. /mnt/huggingface_models):"
echo "If you don't specify a path, the model will be saved in the directory: /mnt/huggingface_models"
read -p "Path to save the model: " model_path
# If the user did not specify a path, then use the default path
if [ -z "$model_path" ]; then
    model_path="/mnt/huggingface_models"
fi

# Create a new download_{timestamp}.yaml file in teh config/automatic/ directory
timestamp=$(date +%s)
config_file="config/automatic/download_${timestamp}.yaml"
# The content of the config file should look like:
#models_to_download:
#  - facebook/incoder-6B
#local_model_folder: /mnt/huggingface_models
echo "models_to_download:" > $config_file
echo "  - $repository" >> $config_file
echo "local_model_folder: $model_path" >> $config_file

# Run the download script
python -m attwizard.script.download_model $config_file