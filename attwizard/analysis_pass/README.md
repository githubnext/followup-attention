# Analysis Pass

With analysis pass we refer to a computation done on the attention weights.
Some non exhaustive examples could include:
1. (yes/no data) whether the produce code was syntactically correct or not.
1. (yes/no data) whether the produce code passed the test suite or not.
1. (numerical data) average attention per token category (e.g. identifier, operators, separators, etc.)
1. (numerical data) attention entropy.
1. (numerical data) number of generated tokens.
1. (numerical data) Jensen-Shannon divergence between attention weights on different layers adn different attention heads, as heatmap.
1. (crispiness of attention weights) array with 100 points, representing how fast the attention decreases between the most attended token and the least attended ones, a steep decrease gives a peaked attention (i.e. few tokens receive most attention), a slow decrease gives a fairly distributed attention over many tokens.
1. (derived attention measures) such as Attention Flow or Attention Rollout.
1. (derived attention measures) attention weight as transitive attention of the last and second-last layer.
1. (image data) heatmap over source code for the attention weights of the first token in the prediction.
1. (image data) heatmap over source code for the attention weights of specific token sequences (e.g. `):` end of a function signature in python).


## Start the Analysis (i.e. derive new machine data)
To run the analysis pass you have to run the following command:

```bash
python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml deriveall
```
The script will transform the attention tensor according to the strategies metnioned in the key `analysis_passes`. They will all be triggered, comment them out if you want to call only a subset.
For convenence the `analysis_data_folder` is the same of the `output_data_folder` since they are connected to the Azure Storage container and do not use the current machine memory, and it is easier to maintain a single container per experiment.
Check `exp_v08.yaml` for an example of configuration file.


Remember that you can always prefix the command with `screen -L -Logfile data/model_output/exp_vXX/log.txt ...` to run the analysis pass in the background.

## Compare the Data (e.g. Human vs Machine)

To compre human vs machine data you can use the following command:

```bash
python -m attwizard.analysis_pass.comparator --config config/comparisons/cmp_vXX.yaml compare
```

This compares data as specified in the `comparisons` key of the configuration file and generates a `.csv` file for each comparison.
Note that this comparison functions work both for vectors and matrices, moreover you can also apply custom chains of transformations to the data before the comparison (using the `transformations` subfield of a `comparison`); for example you might want to normalize the data to look like probabilities.
For a full example you can have a look at `cmp_v01.yaml`.


## Grade Answers

To extract the machine answers from the generated text use the twofollowing commands:
```bash
python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml getanswers
python -m attwizard.analysis_pass.analyzer --config config/exp_vXX.yaml getanswersdataset
```
This will create a `.csv` file with the machine answers in the `output_data_folder` specified in the configuration file.
it will focus on the first model specified in the `models` key of the configuration file.
For a full example you can have a look at `exp_v10.yaml`.