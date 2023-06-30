from hashlib import new
from attwizard.creators.prompt_creator import PromptCreator

from typing import List, Dict, Any, Tuple
from attwizard.creators.dep_graph import DepNode, AssignmentOp, Operator, QueryOp, InfixOp, Program
from attwizard.creators.distance_to_weights import convert_distance_to_weight_naive
from random import randint
from random import choice, sample
from itertools import combinations
import os
import numpy as np
import random
from tqdm import tqdm


class TwoFunctionsOneTestCreator(PromptCreator):

    def __init__(
            self,
            name: str,
            prompt_folder: str, ground_truth_folder: str,
            n_prompts: int = 10,
            attention_level: str = 'line', tokenizer=None,
            kwargs: Dict[str, Any] = None) -> None:
        all_functions = self._get_all_functions(
            folder_with_seed_functions=kwargs["folder_with_seed_functions"],
            start_of_test_statement=kwargs["start_of_test_statement"])
        self.function_pairs = list(combinations(all_functions, 2))
        # shuffle the function pairs
        random.shuffle(self.function_pairs)
        self.next_index_to_pick = 0
        super().__init__(name, prompt_folder, ground_truth_folder,
                         n_prompts, attention_level, tokenizer, kwargs)

    def _get_all_functions(
            self,
            folder_with_seed_functions: str,
            start_of_test_statement: str) -> List[Dict[str, str]]:
        """Get the functions in the folder possible functions."""
        all_files = os.listdir(folder_with_seed_functions)
        py_files = [f for f in all_files if f.endswith(".py")]
        functions = []
        print(
            f"Found {len(py_files)} python" +
            f" files in {folder_with_seed_functions}")
        print("Reading functions...")
        for filename in tqdm(py_files):
            content = open(
                os.path.join(
                    folder_with_seed_functions, filename), 'r').read()
            lines = content.split("\n")
            start_test = False
            lines_for_function = []
            lines_for_test = []
            for line in lines:
                if line.startswith(start_of_test_statement):
                    start_test = True
                if not start_test:
                    lines_for_function.append(line)
                else:
                    lines_for_test.append(line)
            section_function = "\n".join(lines_for_function)
            section_test = "\n".join(lines_for_test)
            functions.append({
                    "filename": filename,
                    "function": section_function,
                    "test": section_test
            })
        return functions

    def create_prompt_and_attention(self) -> Tuple[List[str], List[float]]:
        """Create a new prompt and attention.

        In this case, each part of prompt is a single line.
        """
        # get a random function pair
        function_pair = self.function_pairs[self.next_index_to_pick]
        self.next_index_to_pick += 1
        # get the two function and test
        first_function = function_pair[0]["function"]
        second_function = function_pair[1]["function"]
        # pick one test
        target_test_index = choice([0, 1])
        target_test = function_pair[target_test_index]["test"]
        # create the prompt
        prompt_parts = first_function.split("\n")
        prompt_parts.extend([""])
        prompt_parts.extend(second_function.split("\n"))
        prompt_parts.extend([""])
        prompt_parts.extend(target_test.split("\n"))
        # create the attention
        if target_test_index == 0:
            att_weights = \
                [1.0] * len(first_function.split("\n")) + [0] + \
                [0] * len(second_function.split("\n")) + [0] + \
                [1.0] * len(target_test.split("\n"))
        elif target_test_index == 1:
            att_weights = \
                [0] * len(first_function.split("\n")) + [0] + \
                [1.0] * len(second_function.split("\n")) + [0] + \
                [1.0] * len(target_test.split("\n"))
        return prompt_parts, att_weights
