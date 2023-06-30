#!/bin/bash

# WARNING: this script is long running thus run it in a screen session
# use the command:
# screen -S master_queryderive -m ./2_QUERY_MODEL.sh

# Query for which experiment template to use among the available templates
# in folder: config/template/experiments
avilable_templates=$(ls config/template/experiments)
# assign a number to each template
declare -A templates
i=1
for template in $avilable_templates; do
    templates[$i]=$template
    i=$((i+1))
done
# print the templates
echo "Available experiment templates:"
for key in "${!templates[@]}"; do
    echo "$key) ${templates[$key]}"
done
# ask the user to choose a template
read -p "Choose a template: " template_number
# check if the user input is valid
if [ -z "${templates[$template_number]}" ]; then
    echo "Invalid input. Please choose a number between 1 and ${#templates[@]}"
    exit 1
fi
# assign the chosen template to a variable
template=${templates[$template_number]}
echo "You chose the template: $template"
# get the filename of the chosen template without extension
template_filename_without_extension=$(basename $template .yaml)

# Query for the name of the huggingface repository
echo "Enter the name of the model you want to query (e.g. facebook/incoder-6B):"
read -p "Model Name as Huggingface repository name: " repository
# check if the user input is valid. it should have a slash
if [[ ! $repository == *"/"* ]]; then
    echo "Invalid input. Please enter a valid Huggingface repository name"
    exit 1
fi


# ask for a short name of the model to use in the experiment name
echo "Enter a short name for the model (e.g. incoder-6B):"
read -p "Short name for the model: " model_name
# check if the user input is valid. It should be not emptu and there should be no space.
if [ -z "$model_name" ] || [[ $model_name == *" "* ]]; then
    echo "Invalid input. Please enter a valid short name for the model (no space and not empty)"
    exit 1
fi

# Ask to specify where the model is stored, otherwise the default
# location is the directory: /mnt/huggingface_models
echo "Enter the path where the model is stored (e.g. /mnt/huggingface_models):"
echo "If you don't specify a path, /mnt/huggingface_models will be used"
read -p "Path to save the model: " model_path
# If the user did not specify a path, then use the default path
if [ -z "$model_path" ]; then
    model_path="/mnt/huggingface_models"
fi

# Create a new query_{timestamp}.yaml file in the config/automatic/ directory
timestamp=$(date +%s)
huaman_readable_timestamp_no_space=$(date +%Y-%m-%d_%H-%M-%S)
config_file="config/automatic/${template_filename_without_extension}_${model_name}_${huaman_readable_timestamp_no_space}.yaml"
# copy the template to the new config file location
cp config/template/experiments/$template $config_file

# Replace the jinja variables in the config file
# {{model_huggingface_name}} -> repository
# {{local_model_folder}} -> model_path
# {{model_short_name}} -> model_name
# before sed escape the slashes in the repository name and the model_path
repository_escaped=$(echo $repository | sed 's/\//\\\//g')
model_path_escaped=$(echo $model_path | sed 's/\//\\\//g')
sed -i "s/{{model_huggingface_name}}/$repository_escaped/g" $config_file
sed -i "s/{{local_model_folder}}/$model_path_escaped/g" $config_file
sed -i "s/{{model_short_name}}/$model_name/g" $config_file

# Show the first 10 lines of the config file
echo "The first 10 lines of the config file are:"
echo "-------------------------------------------------------------------------"
head -n 10 $config_file
echo "..."
echo "-------------------------------------------------------------------------"
echo "The full config file is located at: $config_file"

# Run the query script in Screen
# create a log path:
log_path="config/automatic/logs/${template_filename_without_extension}_${model_name}_${huaman_readable_timestamp_no_space}.txt"
echo "The log file is located at: $log_path"
screen -mS query_${model_name}_${timestamp} -L -Logfile ${log_path} bash -c "python -m attwizard.script.batch_attention_extraction --config ${config_file} queryextract"
# python -m attwizard.script.batch_attention_extraction --config $config_file queryextract

# Derive the answers
#python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml getanswers
#python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml getanswersdataset
screen -mS answers_${model_name}_${timestamp} -L -Logfile ${log_path} bash -c "python -m attwizard.analysis_pass.analyzer --config ${config_file} getanswers"
screen -mS answersdataset_${model_name}_${timestamp} -L -Logfile ${log_path} bash -c "python -m attwizard.analysis_pass.analyzer --config ${config_file} getanswersdataset"

# Run the post_processing script to derive attention
# python -m attwizard.analysis_pass.analyzer --config config/exp_v10_incoder.yaml deriveall
screen -mS deriveall_${model_name}_${timestamp} -L -Logfile ${log_path} bash -c "python -m attwizard.analysis_pass.analyzer --config ${config_file} deriveall"