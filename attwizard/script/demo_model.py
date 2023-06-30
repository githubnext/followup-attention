from typing import Dict, List, Tuple, Union, Any

from attwizard.script.utils_model import load_tokenizer_and_model
from attwizard.script.utils_model import query_which_model
from attwizard.script.utils_model import run_model_single_instance
from attwizard.script.utils import get_prompt
from attwizard.script.utils import read_config_file


def main_loop(config: Dict[str, str]):
    """
    Main loop for the demo.
    """
    picked_model_name = query_which_model(config["models_to_download"])
    model, tokenizer = load_tokenizer_and_model(
        config, model_name=picked_model_name)
    while True:
        prompt = get_prompt()
        run_model_single_instance(
            model, tokenizer, prompt)


if __name__ == "__main__":
    config = read_config_file("config/download_config.yaml")
    main_loop(config)
