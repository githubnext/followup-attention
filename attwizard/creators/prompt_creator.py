"""Base class for generating new prompts."""

from abc import ABC, abstractmethod

from typing import List, Dict, Any, Tuple
import os
import json
import random
import numpy as np
import torch


class PromptCreator(ABC):

    def __init__(
            self,
            name: str,
            prompt_folder: str, ground_truth_folder: str,
            n_prompts: int = 10,
            attention_level: str = 'line',
            tokenizer=None,
            kwargs: Dict[str, Any] = None) -> None:
        super().__init__()
        self.name = name
        self.prompt_folder = prompt_folder
        self.ground_truth_folder = ground_truth_folder
        self.n_prompts = n_prompts
        self.attention_level = attention_level
        self.tokenizer = tokenizer
        self.kwargs = kwargs if kwargs else {}
        # initialize internal state
        self.prompts = []
        self.prompt_names = []
        self.prompt_attention = []

    def create_prompts(self) -> None:
        """Create new prompts and attention."""

        if "seed" in self.kwargs.keys():
            seed = self.kwargs["seed"]
            # set the python seed for reproducibility
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        for i in range(self.n_prompts):
            i_name = f"prompt_{i}_{self.name}"
            self.prompt_names += [i_name]
            prompt_parts, att_weights = self.create_prompt_and_attention()
            # aggregate the prompt_parts
            if self.attention_level == "line":
                prompt = "\n".join(prompt_parts)
            elif self.attention_level == "token":
                prompt = "".join(prompt_parts)
            # save the prompt
            self.prompts.append(prompt)
            with open(os.path.join(self.prompt_folder, i_name + ".txt"), 'w') as f:
                f.write(prompt)
            # save the attention
            self.prompt_attention.append(att_weights)
            metadata = {}
            metadata["attendable_parts"] = prompt_parts
            metadata["attention_level"] = self.attention_level
            metadata["attention_weights"] = att_weights
            metadata["args"] = self.kwargs
            # class name of the creator
            metadata["creation_strategy"] = self.__class__.__name__
            # dump the dictionary as a json file
            with open(os.path.join(self.ground_truth_folder, i_name + ".json"), 'w') as f:
                json.dump(metadata, f)
            print(f"Created prompt {i_name}")


    @abstractmethod
    def create_prompt_and_attention(self) -> Tuple[List[str], List[float]]:
        """Create a single new prompt and attention.

        This function is called n_prompts times.
        It should return a tuple containing a single prompt and attention.
        The prompt is given as a sequence of "attendable parts" (e.g. tokens or
        lines). The attention is given as a sequence of weights for each
        "attendable part".
        """
        pass
