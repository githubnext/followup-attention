from attwizard.creators.prompt_creator import PromptCreator

from typing import List, Dict, Any

class GibberishPromptCreator(PromptCreator):

    def __init__(
            self, name: str,
            prompt_folder: str, ground_truth_folder: str,
            n_prompts: int = 10, kwargs: Dict[str, Any] = None) -> None:
        super().__init__(name, prompt_folder, ground_truth_folder, n_prompts, kwargs)

    def create_prompt(self) -> None:
        """Create new prompts according to the given strategies."""
        for i in range(self.n_prompts):
            self.prompts.append(self.create_prompt_i())
            self.prompt_names.append(f"{self.name}_{i}")
            self.prompt_attention.append(self.create_expected_attention_ground_truth_i())

    def create_prompt_i(self) -> str:
        """Create new prompts according to the given strategies."""
        return "".join(self.create_prompt_i_char())

    def create_prompt_i_char(self) -> List[str]:
        """Create new