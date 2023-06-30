#!/bin/bash

# WARNING: this script is long running thus run it in a screen session
# use the command:
# screen -S master_interaction_matrix_and_compare -m ./3_COMPARE_WITH_HUMAN.sh

# Query for which experiment template to use among the available templates
# in folder: config/template/comparisons
avilable_templates=$(ls config/template/comparisons)
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
echo "Enter the name of teh model you want to query (e.g. facebook/incoder-6B):"
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
read -p "Path to save the model (enter to use default: /mnt/huggingface_models): " model_path
# If the user did not specify a path, then use the default path
if [ -z "$model_path" ]; then
    model_path="/mnt/huggingface_models"
fi

# Create a new query_{timestamp}.yaml file in the config/automatic/ directory
timestamp=$(date +%s)
huaman_readable_timestamp_no_space=$(date +%Y-%m-%d_%H-%M-%S)
cmp_config_file="config/automatic/${template_filename_without_extension}_${model_name}_${huaman_readable_timestamp_no_space}.yaml"
# copy the template to the new config file location
cp config/template/comparisons/$template $cmp_config_file


# Query for which experiment template to use among the available templates
# in folder: config/template/eye this will determine which eye_config to use
avilable_templates=$(ls config/template/eye)
# assign a number to each template
declare -A templates
i=1
for template in $avilable_templates; do
    templates[$i]=$template
    i=$((i+1))
done
# print the templates
echo "Available eye templates:"
for key in "${!templates[@]}"; do
    echo "$key) ${templates[$key]}"
done
# ask the user to choose a template
read -p "Choose a template (for the post-processing of the eye tracking data): " template_number
# check if the user input is valid
if [ -z "${templates[$template_number]}" ]; then
    echo "Invalid input. Please choose a number between 1 and ${#templates[@]}"
    exit 1
fi
# assign the chosen template to a variable
template_eye=${templates[$template_number]}
echo "You chose the template: $template_eye"
# get the filename of the chosen template without extension
template_eye_filename_without_extension=$(basename $template_eye .yaml)

# Create a new yaml file in the config/automatic/ directory
eye_config_file="config/automatic/${template_eye_filename_without_extension}_${model_name}_${huaman_readable_timestamp_no_space}.yaml"
# copy the template to the new config file location
cp config/template/eye/$template_eye $eye_config_file

# ask the user which decay to use, otherwise the deafult value 0.1 is used
# the decay must be a float number otherwise fail
echo "Enter the decay value (e.g. 0.1):"
read -p "Decay value (enter to use default: 0.1): " decay
# If the user did not specify a decay, then use the default decay
if [ -z "$decay" ]; then
    decay="0.1"
fi
# check if the decay is a float number
if ! [[ $decay =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid input. Please enter a valid decay value (a float number)"
    exit 1
fi

# Replace the jinja variables in the config file
# {{model_huggingface_name}} -> repository
# {{local_model_folder}} -> model_path
# {{model_short_name}} -> model_name
# {{model_huggingface_name}} -> repository_name_escaped_slashes
# {{decay}} -> decay
# {{repo_folder}} -> cwd
# before sed escape the slashes in the repository name and the model_path
CWD=$(pwd)
repository_escaped=$(echo $repository | sed 's/\//\\\//g')
model_path_escaped=$(echo $model_path | sed 's/\//\\\//g')
cwd_escaped=$(echo $CWD | sed 's/\//\\\//g')
repository_name_escaped_slashes=${repository//\//_}

# replace all the variable for both config files in a loop
all_config_files=($cmp_config_file $eye_config_file)
for config_file in "${all_config_files[@]}"; do
    sed -i "s/{{model_huggingface_name}}/$repository_escaped/g" $config_file
    sed -i "s/{{local_model_folder}}/$model_path_escaped/g" $config_file
    sed -i "s/{{model_short_name}}/$model_name/g" $config_file
    sed -i "s/{{underscore_model_huggingface_name}}/$repository_name_escaped_slashes/g" $config_file
    sed -i "s/{{decay}}/$decay/g" $config_file
    sed -i "s/{{repo_folder}}/$cwd_escaped/g" $config_file
    # Show the first 10 lines of the config file
    echo "The first 10 lines of the config file $config_file are:"
    echo "-------------------------------------------------------------------------"
    head -n 10 $config_file
    echo "..."
    echo "-------------------------------------------------------------------------"
    echo "The full config file is located at: $config_file"
done

# press to continue
read -p "Press enter to continue"

# Run the query script in Screen
# create a log path:
log_path="config/automatic/logs/${template_eye_filename_without_extension}_${model_name}_${huaman_readable_timestamp_no_space}.txt"
echo "The log file is located at: $log_path"

# Derive the weights from the eye tracking data
# python -m eye_tracking.post_processing config/eye_vXX.yaml getweights
screen -mS getweights_${model_name}_${timestamp} -L -Logfile $log_path bash -c "python -m eye_tracking.post_processing $eye_config_file getweights"


# # Derive the machine asnswers in a dataset
# # python -m eye_tracking.post_processing config/eye_vXX.yaml getfollowup
# # python -m eye_tracking.post_processing config/eye_vXX.yaml getavgbehavior
screen -mS getfollowup_${model_name}_${timestamp} -L -Logfile $log_path bash -c "python -m eye_tracking.post_processing $eye_config_file getfollowup"
screen -mS getavgbehavior_${model_name}_${timestamp} -L -Logfile $log_path bash -c "python -m eye_tracking.post_processing $eye_config_file getavgbehavior"


# Get the attention weights for the model
# python -m eye_tracking.post_processing config/eye_vXX.yaml derivetokenweights
screen -mS derivetokenweights_${model_name}_${timestamp} -L -Logfile $log_path bash -c "python -m eye_tracking.post_processing $eye_config_file derivetokenweights"

log_path="config/automatic/logs/${template_filename_without_extension}_${model_name}_${huaman_readable_timestamp_no_space}.txt"
echo "The log file is located at: $log_path"
# Run the comparison script
# python -m attwizard.analysis_pass.comparator --config config/comparisons/cmp_vXX compare
screen -mS compare_${model_name}_${timestamp} -L -Logfile $log_path bash -c "python -m attwizard.analysis_pass.comparator --config $cmp_config_file compare"