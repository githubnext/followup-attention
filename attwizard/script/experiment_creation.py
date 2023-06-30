"""This file creates new prompt to be used in the experiment."""

import yaml
import click
import random
from pathlib import Path
from typing import List, Dict, Any
import json

from attwizard.creators.assignement_chain import AssignmentChainCreator
from attwizard.creators.two_functions_one_test import TwoFunctionsOneTestCreator


@click.group()
@click.option(
    '--config', default=None,
    help="Pass path to yaml config file with keys: " +
         "prompt_creation_strategy, " +
         "input_data_folder.")
@click.pass_context
def cli(ctx, config):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    # read yaml config file
    if config is not None:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        ctx.obj['CONFIG'] = config
    else:
        click.echo("No config file passed.")
        return


def produce_gibberish_file(
        max_lines: int = 15,
        max_char_per_line: int = 80,
        max_token_length: int = 15,
        space_probability: float = 0.33,
        allowed_characters: str = "abcdefghijklmnopqrstuvwxyz.,[]():;'\"!?-_=+/\\*&%$#@<>|~") -> str:
    """Produce a gibberish file."""
    content = ""

    def get_random_char():
        """Get a random character."""
        return random.choice(allowed_characters)

    def get_random_token(max_token_length: int):
        """Get a random token."""
        return "".join([
            get_random_char()
            for _ in range(int(random.random() * max_token_length))
        ])

    n_lines = 1 + int(random.random() * (max_lines - 1))
    for _ in range(n_lines):
        budget_left = 1 + int(random.random() * (max_char_per_line - 1))
        while budget_left > 0:
            token = get_random_token(max_token_length)
            if (random.random() < space_probability):
                token = " " + token
            if budget_left - len(token) < 0:
                # truncate if it is over the budget for this line
                token = token[:budget_left]
            budget_left -= len(token)
            content += token
        content += "\n"
    return content, None


def produce_assignment_file_and_attention(
        relevant_variable: str = "a",
        supporting_variables: List[str] = ["b", "c"],
        useless_variables: List[str] = ["d", "e", "f"],
        assignemnt_chain_length: int = 3,
        n_useless_operations: int = 10,
        custom_prompt: str = "if (<RELEVANT_VARIABLE> == ",
    ):
    """Create a file with assignments and return the expected attention.

    An example could be:
    ```python
    a = 1
    b = 3
    sum = a + b
    if (sum ==
    ```
    """
    lines = []
    statement_relevance = []
    # define useless variables first
    new_lines = define_variables(useless_variables)
    statement_relevance += [False] * len(new_lines)
    lines += new_lines
    # define supporting variables
    new_lines = define_variables(supporting_variables)
    statement_relevance += [False] * len(new_lines)
    lines += new_lines
    # define relevant variable
    new_lines = define_variables([relevant_variable])
    statement_relevance += [True] * len(new_lines)
    lines += new_lines
    # define assignement chains, then shuffle them before adding them
    lines_to_shuffle = []
    relevance_to_shuffle = []
    # relevant statment
    for _ in range(assignemnt_chain_length):
        if random.random() < 0.5:
            line, relev = random_arithmetic_op(
                assigned_variable=relevant_variable,
                relevant_variables=relevant_variable,
                possible_operands=supporting_variables)
        else:
            line, relev = random_function_call(
                assigned_variable=relevant_variable,
                relevant_variables=relevant_variable,
                possible_arguments=supporting_variables)
        lines_to_shuffle.append(line)
        relevance_to_shuffle.append(relev)
    # useless statments
    for _ in range(n_useless_operations):
        random_assigned_variable = random.choice(
            supporting_variables + useless_variables)
        if random.random() < 0.5:
            line, relev = random_arithmetic_op(
                assigned_variable=random_assigned_variable,
                relevant_variables=relevant_variable,
                possible_operands=supporting_variables + useless_variables,)
        else:
            line, relev = random_function_call(
                assigned_variable=random_assigned_variable,
                relevant_variables=relevant_variable,
                possible_arguments=supporting_variables + useless_variables,)
        lines_to_shuffle.append(line)
        relevance_to_shuffle.append(relev)
    # shuffle the lines and relevance
    shuffable_pairs = list(zip(lines_to_shuffle, relevance_to_shuffle))
    random.shuffle(shuffable_pairs)
    shuffled_lines, relevance_of_shuffled_lines = zip(*shuffable_pairs)
    lines += shuffled_lines
    statement_relevance += relevance_of_shuffled_lines
    # add custom prompt
    custom_prompt = custom_prompt.replace(
        "<RELEVANT_VARIABLE>", relevant_variable)
    lines.append(custom_prompt)
    statement_relevance.append(True)
    assert len(lines) == len(statement_relevance)
    content = "\n".join(lines)
    print(content)
    relevance_metadata = {
        "relevant_variable": relevant_variable,
        "supporting_variables": supporting_variables,
        "useless_variables": useless_variables,
        "attention_level": "statement",
        "attention_weights": statement_relevance,
        "prediction_level": "next_token"
    }
    return content, relevance_metadata


def define_variables(variable_to_define: List[str]):
    """Define the variable with some primitive types."""
    new_statements = []
    for variable in variable_to_define:
        new_statements.append(variable + " = " + random_primitive_type())
    return new_statements


def random_primitive_type(
        possible_strings: List[str] = ['"hello"', '"world"', '"string"'],
        possible_numbers: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        possible_booleans: List[str] = ["True", "False"]):
    """Randomly choose a primitive type."""
    possible_res = possible_strings + possible_numbers + possible_booleans
    return random.choice(possible_res)


def random_function_call(
        assigned_variable: str,
        relevant_variables: str,
        possible_arguments: List[str],
        functions: List[str] = ["abs", "max", "min", "len"],
        max_n_arguments: int = 5,
        prob_primitive_type: float = 0.1):
    """Randomly create a function call."""
    n_args = 1 + int(random.random() * (max_n_arguments - 1))
    used_variables = []
    new_statement = []
    new_statement.append(assigned_variable)
    used_variables.append(assigned_variable)
    new_statement.append("=")
    new_statement.append(random.choice(functions) + "(")
    argument = random.choice(possible_arguments)
    new_statement.append(argument + ",")
    used_variables.append(argument)
    for i in range(n_args - 1):
        if random.random() < prob_primitive_type:
            argument = random_primitive_type()
        else:
            argument = random.choice(possible_arguments)
        new_statement.append(argument + ",")
        used_variables.append(argument)
    new_statement.append(")")
    relevant_statement = any([
        relevant_var in used_variables
        for relevant_var in relevant_variables])
    statement_line = " ".join(new_statement)
    return statement_line, relevant_statement



def random_arithmetic_op(
        assigned_variable: str,
        relevant_variables: str,
        possible_operands: List[str],
        max_operand_length: int = 5,
        arithmetic_ops: List[str] = ["+", "-", "*", "/"],
        prob_primitive_type: float = 0.1):
    """Randomly choose an arithmetic operation."""
    n_operands = 1 + int(random.random() * (max_operand_length - 1))
    new_statement = []
    new_statement.append(assigned_variable)
    new_statement.append("=")
    new_statement.append(random.choice(possible_operands))
    for i in range(n_operands - 1):
        # random arithmetic operation
        new_statement.append(random.choice(arithmetic_ops))
        # random supporting variable
        if random.random() < prob_primitive_type:
            operand = random_primitive_type()
        else:
            operand = random.choice(possible_operands)
        new_statement.append(operand)
    relevant_statement = any([
        relevant_var in new_statement
        for relevant_var in relevant_variables])
    print(new_statement)
    statement_line = " ".join(new_statement)
    return statement_line, relevant_statement




@cli.command()
@click.pass_context
def create(ctx):
    """Create new prompts according to different strategies."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)
    prompt_folder = config.get('input_data_folder', None)
    # make sure the output folder exists
    Path(prompt_folder).mkdir(parents=True, exist_ok=True)
    # if there are files already ask if they should be overwritten
    if len(list(Path(prompt_folder).glob("*.txt"))) > 0:
        click.echo("There are already files in the folder. Overwrite?")
        if not click.confirm("Do you want to overwrite?"):
            click.echo("Aborting.")
            return
    strategy = config.get('prompt_creation_strategy', None)
    click.echo(f"Creating prompts with {strategy['name']} strategy.")

    if strategy['name'] == 'complete_gibberish':
        creation_fn = produce_gibberish_file
    if strategy['name'] == 'assignment_chain':
        creation_fn = produce_assignment_file_and_attention

    for i in range(strategy['n_prompts']):
        prompt_filename = f"prompt_{i}.txt"
        prompt_path = Path(prompt_folder, prompt_filename)
        content, metadata = creation_fn(**strategy['kwargs'])
        with open(prompt_path, 'w') as f:
            f.write(content)
        with open(prompt_path.with_suffix(".json"), 'w') as f:
            json.dump(metadata, f)
        click.echo(f"Created prompt {prompt_path}.")


@cli.command()
@click.pass_context
def creator(ctx):
    """Create new prompts according to different Creator strategies."""
    ctx.ensure_object(dict)
    config = ctx.obj.get('CONFIG', None)
    prompt_folder = config.get('input_data_folder', None)
    ground_truth_folder = config.get('ground_truth_folder', None)
    # make sure the output folder exists
    Path(prompt_folder).mkdir(parents=True, exist_ok=True)
    Path(ground_truth_folder).mkdir(parents=True, exist_ok=True)
    # if there are files already ask if they should be overwritten
    if len(list(Path(prompt_folder).glob("*.txt"))) > 0:
        click.echo("There are already files in the folder. Overwrite?")
        if not click.confirm("Do you want to overwrite?"):
            click.echo("Aborting.")
            return
    strategy = config.get('prompt_creation_strategy', None)
    click.echo(f"Creating prompts with {strategy['name']} strategy.")

    assert strategy['type'] == 'creator'

    if strategy['name'] == 'assignment_chain':
        creator = AssignmentChainCreator(
            name=strategy['name'],
            prompt_folder=prompt_folder,
            ground_truth_folder=ground_truth_folder,
            n_prompts=strategy['n_prompts'],
            attention_level=strategy['attention_level'],
            kwargs=strategy['kwargs'])

    if strategy['name'] == 'two_functions_one_test':
        creator = TwoFunctionsOneTestCreator(
            name=strategy['name'],
            prompt_folder=prompt_folder,
            ground_truth_folder=ground_truth_folder,
            n_prompts=strategy['n_prompts'],
            attention_level=strategy['attention_level'],
            kwargs=strategy['kwargs'])

    creator.create_prompts()



if __name__ == '__main__':
    cli()
