## Overview

This folder contains the routine to create synthetic prompt files with specific structures.

Each creator generates both prompts and ground truth on the expected attention patterns.

The attention pattern can be:
- (READY) line level (e.g. regardless of the break lines of the same statement)
- (TODO) token-level (e.g those of python tokenizer or those of gpt-2 tokenizer)
- (TODO) function level

Express the destination of your computation via the config file:
- `input_data_folder`: the folder where the prompt snippets are stored.
- `ground_truth_folder`: the folder where the ground truth attention weights and metadata are stored.

There are two types of generations:
- those in which we do not expect any particular attention pattern (e.g gibberish text)
- those in which we expect a particular attention pattern (e.g. successive assignment)

## Test Suite

To run the test suite run this command from the root folder of the repository:
```bash
python -m pytest -s attwizard/creators/test_assignment_chain.py
```