import os
import yaml
import numpy as np
import torch
from torch import nn
from os.path import join
from pathlib import Path
from transformers import AutoTokenizer, GPTNeoForCausalLM, GPTJForCausalLM, AutoModelForCausalLM
from attwizard.models.modeling_codegen import CodeGenForCausalLM
from typing import Dict, List, Tuple, Union, Any

from attwizard.script.utils import query_which_model
from attwizard.script.utils import get_model_folder_path
from attwizard.script.utils import load_json_file
from attwizard.aligner import get_tokens_with_col_and_line


def get_model_tokenization(
        raw_text: str,
        tokenizer: str,
        model_folder: str,
        separate_new_lines_tokens: bool = True):
    """Create the tokenization of the raw_text.

    The tokenization format as output is the one used to plot it in the
    codeattention library.
    """
    hugging_short_repo = tokenizer
    model_folder_path = get_model_folder_path(
        model_folder=model_folder,
        hugging_short_repo=hugging_short_repo
    )
    tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

    tmp = tokenizer(raw_text, return_tensors="pt")
    tokens = tmp['input_ids'][0].tolist()
    # model specific hack
    if "incoder" in hugging_short_repo:
        # drop the first token
        tokens = tokens[1:]
    # convert the ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    # replace spaces and new lines
    tokens = [
        token.replace("Ċ", "\n").replace('Ġ', ' ') for token in tokens]
    # add the start line position
    formatted_tokens = []
    line = 1
    column = 1
    for i, token in enumerate(tokens):
        if separate_new_lines_tokens and any([token == "\n" * r for r in range(10)]):
            for _ in range(token.count("\n")):
                formatted_tokens.append({
                    "t": "\n",
                    "i": i,
                    "l": line,
                    "c": column}
                )
                line += 1
                column = 1
        else:
            formatted_tokens.append({
                "t": token,
                "i": i,
                "l": line,
                "c": column}
            )
            column += len(token)
            if not separate_new_lines_tokens and "\n" in token:
                line += token.count("\n")
    return formatted_tokens


def load_tokenizer_and_model(config: Dict[str, str], model_name: str):
    """
    Loads a model from the local huggingface repository.
    """
    model_folder_path = get_model_folder_path(
        model_folder=config["local_model_folder"],
        hugging_short_repo=model_name
    )
    if "gpt-neo" in model_name:
        model = GPTNeoForCausalLM.from_pretrained(model_folder_path)
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
    elif "gpt-j" in model_name:
        model = GPTJForCausalLM.from_pretrained(
            model_folder_path,
            revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    elif "codegen" in model_name:
        model = CodeGenForCausalLM.from_pretrained(model_folder_path)
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        if "codegen-16B-multi" in model_name or \
                "codegen-6B-multi" in model_name or \
                "codegen-6B-nl" in model_name or \
                "codegen-2B-multi" in model_name:
            # Need to initialize RPC framework first.
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # model = model.to(device)tokenizer.decode
            print("Before calling model.eval()")
            model.eval()
            model.parallelize()
            print("After calling model.parallelize()")
        else:
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
    elif "incoder" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_folder_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_folder_path,
            kwargs={"clean_up_tokenization_spaces": False})
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        raise ValueError("Unknown model name.")

    return model, tokenizer


def run_model_single_instance(
        model, tokenizer, prompt: str,
        n_new_tokens: int = 50,
        max_time_in_seconds: int = -1,
        temperature: float = 1.0,
        seed: int = 42):
    """Run inference with the model and the current prompt."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = tokenizer(prompt, return_tensors="pt")
    input_ids = tmp['input_ids'].to(device)
    attention_mask = tmp['attention_mask'].to(device)
    extra_args = {}
    if max_time_in_seconds > 0:
        extra_args["max_time"] = max_time_in_seconds
    if n_new_tokens > 0:
        extra_args["max_length"] = len(input_ids[0]) + n_new_tokens
    else:
        extra_args["pad_token_id"] = tokenizer.eos_token_id
    torch.manual_seed(seed)
    print("Before calling model.generate()")
    model_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        return_dict_in_generate=True,
        output_attentions=True,
        **extra_args
    )
    print("After calling model.generate()")
    assert len(model_output["sequences"]) == 1, \
        "This method is for inference on a single sequence (not batches)."
    output_ids_cpu = model_output["sequences"][0].detach().cpu()
    input_ids_cpu = input_ids[0].detach().cpu()
    generated_text = tokenizer.decode(output_ids_cpu)
    print(generated_text)
    metadata = {}
    metadata["tokens_all"] = tokenizer.convert_ids_to_tokens(output_ids_cpu)
    metadata["tokens_prompt"] = tokenizer.convert_ids_to_tokens(input_ids_cpu)
    metadata["tokens_generated"] = \
        metadata["tokens_all"][len(metadata["tokens_prompt"]):]
    metadata["text_generated"] = generated_text
    metadata["text_prompt"] = prompt
    return model_output, metadata


def read_model_output(folder: str):
    """Read the result of each code generation task.

    Return a list of dictionaries, each with the following fields:
        - "metadata": the content of the json file of metadata
        - "matrix": the numpy matrix
        - "name": the name of the file
    """
    unique_file_stems = [
        str(p.stem) for p in list(Path(folder).glob("*.json"))]
    records = []
    for file_stem in unique_file_stems:
        new_record = {}
        new_record["name"] = file_stem
        new_record["metadata"] = load_json_file(
            os.path.join(folder, file_stem + ".json"))
        new_record["matrix"] = np.load(
            os.path.join(folder, file_stem + ".npy"))

        records.append(new_record)
    return records


def get_tokenization_from_machine_metadata(
        machine_metadata: Dict[str, Any],):
    """Extract relevant information about the tokenization from the metadata.

    It returns a dictionary with the following:
    {
        "raw_text": the raw text of the prompt
        "clean_tokens_prompt": the list of tokens in the prompt (no special chars)
        "tokenization": the list of records
            [{'s': 0, 'i': 0, 'c': 0, 'l': 0, 't': '/'}, ...
    }
    """

    # derive the original tokenization.
    # we need it because we do not know to which token/line a column belongs.
    raw_text = machine_metadata["text_prompt"]
    model_name = machine_metadata["mode_name"]
    config = machine_metadata["config_options"]
    # get the vocab
    model_folder_path = get_model_folder_path(
        model_folder=config["local_model_folder"],
        hugging_short_repo=model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
    input_ids = tokenizer(raw_text, return_tensors="pt")['input_ids'][0]
    tokens = [tokenizer.decode([i]) for i in input_ids]
    tokenization = get_tokens_with_col_and_line(
        text=raw_text,
        tokens=tokens)
    return {
        "raw_text": raw_text,
        "clean_tokens_prompt": tokens,
        "tokenization": tokenization
    }
