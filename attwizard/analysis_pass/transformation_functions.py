"""Transformations to be applied on the go before the comparisons.

These functions take the two data from human and machine (e.g. vectors or
matrices) and the two respective metadata containing data such as the token
information.
They output a transformed version of the data (e.g. new vectors or matrices).
Note that the shape is not guaranteed to be the same as the original input.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from scipy.sparse import coo_matrix

from attwizard.aligner import map_to_char_level
from attwizard.aligner import get_tokens_with_col_and_line
from attwizard.shaper import aggregate_dim_tokens_to_line


global df_human_cached
global df_machine_chached
df_human_cached = None
df_human_cached = None


def get_transf_fn(name: str) -> Callable:
    """Get the transformation function for the given name.

    Note that this is used for safety reasons to avoid arbitrary code execution
    (e.g. by using eval on the str name of the function).
    """
    # FOR VECTORS
    if name == "convert_vect_to_probabilities":
        return convert_vect_to_probabilities
    elif name == "convert_to_char_level":
        return convert_to_char_level
    # FOR MATRICES
    elif name == "keep_only_tokens_seen_by_model":
        return keep_only_tokens_seen_by_model
    elif name == "normalize_both_matrices_by_line":
        return normalize_both_matrices_by_line
    elif name == "keep_lower_triangle":
        return keep_lower_triangle
    elif name == "remove_specific_tokens":
        return remove_specific_tokens
    elif name == "keep_only_last_layer_machine":
        return keep_only_last_layer_machine
    elif name == "override_machine_with_lower_triangular_of_ones":
        return override_machine_with_lower_triangular_of_ones
    elif name == "keep_only_lines_seen_by_model":
        return keep_only_lines_seen_by_model
    elif name == "convert_to_token_to_lines":
        return convert_to_token_to_lines
    elif name == "remove_base_human_behavior":
        return remove_base_human_behavior
    elif name == "drop_first_model_token":
        return drop_first_model_token
    elif name == "truncate_human_visible_tokens":
        return truncate_human_visible_tokens
    elif name == "norm_by_token_length_human":
        return norm_by_token_length_human
    elif name == "at_least_one_letter":
        return at_least_one_letter
    else:
        raise ValueError("Unknown transformation function: {}".format(name))


def convert_vect_to_probabilities(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the attention weights to probabilities.

    Returns tuple with transformed human data and machine data.
    """
    assert human_data.ndim <= 1, "The convertion to probability using " \
        "`convert_vect_to_probabilities` is supported only for vector input."
    human_data = human_data / np.sum(human_data)
    machine_data = machine_data / np.sum(machine_data)
    return human_data, machine_data


def truncate_human_visible_tokens(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],):
    """Truncate the comparison to the human visible tokens aka prompt.

    It works for vectors only.
    """
    assert human_data.ndim == 1, "The truncation to human visible tokens " \
        "using `truncate_human_visible_tokens` is supported only for vector input."
    n_human_tokens = len(human_data)
    machine_data = machine_data[:n_human_tokens]
    return human_data, machine_data


def drop_first_model_token(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],):
    """Drop the elements of the machine data corresponding to the first token.

    Note that this is done when the model inserts an artificial token that was
    not visible to the human, such as what incoder model does by inserting
    a special <|endoftext|> token at the beginning of the text.

    If a vector, we remove the first element.
    If a matrix we remove first row and first column.
    """
    if human_data.ndim == 1:
        machine_data = machine_data[1:]
    elif human_data.ndim == 2:
        machine_data = machine_data[1:, 1:]
    return human_data, machine_data


def norm_by_token_length_human(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],):
    """Normalize the human attention weights by the length of the token.

    Note that this works only for vectors.
    """
    assert human_data.ndim == 1, "The normalization by token length " \
        "using `norm_by_token_length_human` is supported only for vector input."
    tokens = machine_metadata["tokens_prompt"]
    # remove first token if it is "<|endoftext|>"
    if tokens[0] == "<|endoftext|>":
        tokens = tokens[1:]
    # truncate to the number of machine tokens to match the shape of the human
    n_human_tokens = len(human_data)
    tokens = tokens[:n_human_tokens]
    tokens_lengths = [len(t) for t in tokens]
    # truncate the human data to match the number of machine tokens available
    # in the prompt, typically it includes up to "# Answers:" icnluded
    prompt_tokens_from_machine = tokens
    human_data = human_data[:len(prompt_tokens_from_machine)]
    machine_data = machine_data[:len(prompt_tokens_from_machine)]
    human_data = human_data / tokens_lengths
    return human_data, machine_data


def at_least_one_letter(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],):
    """Keep only weight correctonding to tokens with at least one letter.

    Note that this works only for vectors.
    """
    assert human_data.ndim == 1, "The normalization by token length " \
        "using `at_least_one_letter` is supported only for vector input."
    tokens = machine_metadata["tokens_prompt"]
    # remove first token if it is "<|endoftext|>"
    if tokens[0] == "<|endoftext|>":
        tokens = tokens[1:]
    # truncate to the number of machine tokens to match the shape of the human
    n_human_tokens = len(human_data)
    tokens = tokens[:n_human_tokens]
    # truncate the human data to match the number of machine tokens available
    # in the prompt, typically it includes up to "# Answers:" icnluded
    prompt_tokens_from_machine = tokens
    human_data = human_data[:len(prompt_tokens_from_machine)]
    machine_data = machine_data[:len(prompt_tokens_from_machine)]
    alpha_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    indices_to_keep = [
        i for i, t in enumerate(tokens)
        if any(c in alpha_chars for c in t)]
    human_data = human_data[indices_to_keep]
    machine_data = machine_data[indices_to_keep]
    return human_data, machine_data


def convert_to_char_level(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        mapping_from_token_to_char: str,
        special_char_mapping: Dict[str, str] = None,
        drop_first_token: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the attention data to char level.

    Returns tuple with transformed human data and machine data.
    Note that it works only with vector input.
    """
    assert human_data.ndim <= 1, "The convertion to the char-level is only " \
        "supported for vector input."
    if not special_char_mapping:
        special_char_mapping = {
            "Ġ": ' ',
            "Ċ": '\n',
        }
    machine_tokens = machine_metadata["tokens_prompt"]
    machine_prompt = machine_metadata["text_prompt"]
    if drop_first_token:
        to_remove = machine_tokens[0]
        machine_tokens = machine_tokens[1:]
    machine_vector_char = map_to_char_level(
        tokens=machine_tokens,
        att_weights=list(machine_data),
        raw_text=machine_prompt,
        distribution_fn=mapping_from_token_to_char,
        special_char_mapping=special_char_mapping
    )
    print("machine_vector_char sum: ", np.sum(machine_vector_char))
    # COMPUTE CORRELATION
    # since the model predicts after the `# Answer:` token
    # also the relevant human tokens do not include anything after that
    # we remove everything after the last `:` character
    original_prompt = human_metadata["raw_text"]
    pos_end_of_answer_token = original_prompt.rfind("Answer") + len("Answer")
    human_vector = human_data[:pos_end_of_answer_token]
    machine_vector = machine_vector_char[:pos_end_of_answer_token]
    return human_vector, machine_vector


def convert_to_binary(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the attention weights to binary.

    Returns tuple with transformed human data and machine data.
    """
    human_data = human_data > threshold
    machine_data = machine_data > threshold
    return human_data, machine_data


# MATRIX OPERATIONS


def normalize_both_matrices_by_line(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_prompt: List[str] = None,
        text_prompt: str = None,) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the matrices by line."""
    transformed_data = []
    for matrix in [human_data, machine_data]:
        summation_vector = matrix.sum(axis=1, keepdims=True)
        # replace nan with 1
        summation_vector = np.nan_to_num(summation_vector, nan=1)
        # replace 0 with 1
        summation_vector[summation_vector == 0] = 1
        res = matrix / summation_vector
        res = np.nan_to_num(res, nan=0)
        transformed_data.append(res)
    return transformed_data[0], transformed_data[1]


def convert_to_token_to_lines(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_prompt: List[str] = None,
        text_prompt: str = None,
        special_char_mapping: Dict[str, str] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the matrices to token to lines."""
    if not tokens_prompt:
        tokens_prompt = machine_metadata["tokens_prompt"]
        if tokens_prompt[0] == "<|endoftext|>":
            tokens_prompt = tokens_prompt[1:]
    if not text_prompt:
        text_prompt = machine_metadata["text_prompt"]
        if text_prompt.startswith("<|endoftext|>"):
            text_prompt = text_prompt[len("<|endoftext|>"):]
    # replace all the char with the special char mapping dictioanry
    if special_char_mapping:
        new_tokens_prompt = []
        for t in tokens_prompt:
            new_token = []
            for i, char in enumerate(t):
                if char in special_char_mapping.keys():
                    new_token.append(special_char_mapping[char].replace("\\n", "\n"))
                else:
                    new_token.append(char)
            new_token = "".join(new_token)
            new_tokens_prompt.append(new_token)
        tokens_prompt = new_tokens_prompt
    tokenization = get_tokens_with_col_and_line(text_prompt, tokens_prompt)
    # get the maximum line number
    line_indices = np.array([t["l"] for t in tokenization])
    transformed_data = []
    for att_matrix in [human_data, machine_data]:
        # reduce the attention matrix to the size of the tokenization
        att_matrix = att_matrix[:len(tokenization), :len(tokenization)]
        line_matrix = aggregate_dim_tokens_to_line(
            att_tensor=att_matrix,
            line_indices=line_indices,
            dim=1)
        transformed_data.append(line_matrix)
    return transformed_data[0], transformed_data[1]


def get_prompt_text_tokens_tokenization(
        machine_metadata: Dict[str, Any],):
    """Get the prompt text tokens tokenization."""
    tokens_prompt = machine_metadata["tokens_prompt"]
    text_prompt = machine_metadata["text_prompt"]
    tokenization = get_tokens_with_col_and_line(text_prompt, tokens_prompt)
    return tokenization


def keep_only_lines_seen_by_model(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_prompt: List[str] = None,
        text_prompt: str = None,
        special_char_mapping: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the lines seen by the model (aka the prompt)."""
    # note that the machine has more tokens than the human
    # because it includes the predicted text
    # thus consider only the prompt tokens
    if not tokens_prompt:
        tokens_prompt = machine_metadata["tokens_prompt"]
        if tokens_prompt[0] == "<|endoftext|>":
            tokens_prompt = tokens_prompt[1:]
    if not text_prompt:
        text_prompt = machine_metadata["text_prompt"]
        if text_prompt.startswith("<|endoftext|>"):
            text_prompt = text_prompt[len("<|endoftext|>"):]
    # replace all the char with the special char mapping dictioanry
    if special_char_mapping:
        new_tokens_prompt = []
        for t in tokens_prompt:
            new_token = []
            for i, char in enumerate(t):
                if char in special_char_mapping.keys():
                    new_token.append(special_char_mapping[char].replace("\\n", "\n"))
                else:
                    new_token.append(char)
            new_token = "".join(new_token)
            new_tokens_prompt.append(new_token)
        tokens_prompt = new_tokens_prompt
    tokenization = get_tokens_with_col_and_line(text_prompt, tokens_prompt)
    # get the maximum line number
    max_line = max([t["l"] for t in tokenization])
    n_tokens = len(tokens_prompt)
    # convert to numpy array if not yet
    if not isinstance(machine_data, np.ndarray):
        machine_data = np.array(machine_data)
    if not isinstance(human_data, np.ndarray):
        human_data = np.array(human_data)
    machine_data = machine_data.copy()
    transformed_machine_data = machine_data[:n_tokens, :max_line]
    # normalize by line (so to have probabilities)
    human_data = human_data.copy()
    transformed_human_data = human_data[:n_tokens, :max_line]
    return transformed_human_data, transformed_machine_data


def delete_close_target_lines(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_prompt: List[str] = None,
        text_prompt: str = None,):
    """Delete the lines that are close to current token.

    """
    raise NotImplementedError
    tokenization = get_tokens_with_col_and_line(
        text=text_prompt,
        tokens=tokens_prompt)
    print(tokenization[:20])
    for data in [human_data, machine_data]:
        # iterate over all the starting tokens (aka the lines of data matrix)
        # get line and token position of current token i
        assert min([t['i'] for t in tokenization]) == 0, "tokenization is not 0-based"
        assert min([t['l'] for t in tokenization]) == 0, "line numbering does not start at 0"
        current_line_of_ith_token = tokenization[i]["l"]
        current_token_of_ith_line = tokenization[i]["i"]
        col_before = len(line_human_data)
        if columns_meaning == "token":
            # exclude in the ranking the current token and the tokens above and
            # below it as given by neighbors_to_exclude
            positions_to_exclude = list(range(
                current_line_of_ith_token - neighbors_to_exclude,
                current_line_of_ith_token + neighbors_to_exclude + 1))
            line_human_data = np.delete(line_human_data, positions_to_exclude)
            line_machine_data = np.delete(line_machine_data, positions_to_exclude)
        elif columns_meaning == "line":
            # same for lines
            positions_to_exclude = list(range(
                current_token_of_ith_line - neighbors_to_exclude,
                current_token_of_ith_line + neighbors_to_exclude + 1))
            all_line_indices = set([t["l"] for t in tokenization])
            val_and_line_human = list(zip(list(line_human_data), all_line_indices))
            line_human_data = np.array([
                v for v, l in val_and_line_human
                if l not in positions_to_exclude])
            val_and_line_machine = list(zip(list(line_machine_data), all_line_indices))
            line_machine_data = np.array([
                v for v, l in val_and_line_machine
                if l not in positions_to_exclude])
        col_after = len(line_human_data)
        print(f"line {i}: {col_before} columns -> {col_after} columns")


def keep_only_tokens_seen_by_model(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_prompt: List[str] = None,
        text_prompt: str = None,) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the tokens seen by the model (aka the prompt)."""
    # note that the machine has more tokens than the human
    # because it includes the predicted text
    # thus consider only the prompt tokens
    if not tokens_prompt:
        tokens_prompt = machine_metadata["tokens_prompt"]
        if tokens_prompt[0] == "<|endoftext|>":
            tokens_prompt = tokens_prompt[1:]
    n_tokens = len(tokens_prompt)
    machine_data = machine_data.copy()
    transformed_machine_data = machine_data[:n_tokens, :n_tokens]
    human_data = human_data.copy()
    transformed_human_data = human_data[:n_tokens, :n_tokens]
    return transformed_human_data, transformed_machine_data


def keep_lower_triangle(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the lower triangle of the matrices."""
    assert human_data.shape == machine_data.shape, "The matrices must have " \
        f"the same shape. Human: {human_data.shape}, machine: {machine_data.shape}"
    assert (human_data.shape[0] == human_data.shape[1]
            and machine_data.shape[0] == machine_data.shape[1]), "The matrices " \
        "must be square."
    # replace all the upper triangle with 0
    transformed_machine_data = np.tril(machine_data)
    transformed_human_data = np.tril(human_data)
    return transformed_human_data, transformed_machine_data


def keep_only_last_layer_machine(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only the last layer of the matrices."""
    assert (len(machine_data.shape) == 3), "The matrices must be 3D. " \
        "And the first must be the layer number."
    assert machine_data.shape[1] == machine_data.shape[2], "The matrices " \
        "must be square."
    transformed_machine_data = machine_data[-1, :, :]
    assert len(transformed_machine_data.shape) == 2, "The output transformed" \
        " machine matrix must be 2D."
    return human_data, transformed_machine_data


# FILTERING

def remove_specific_tokens(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_to_remove: str) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the end of line tokens."""
    all_tokens = machine_metadata["tokens_prompt"]
    indices_to_keep = [
        i for i, token in enumerate(all_tokens)
        if token not in tokens_to_remove]
    tr_human_data = human_data[indices_to_keep, :]
    tr_human_data = tr_human_data[:, indices_to_keep]
    tr_machine_data = machine_data[indices_to_keep, :]
    tr_machine_data = tr_machine_data[:, indices_to_keep]

    assert tr_human_data.shape == tr_machine_data.shape, "The matrices must " \
        f"have the same shape. Human: {tr_human_data.shape}, machine: " \
        f"{tr_machine_data.shape}"
    n_to_keep = len(indices_to_keep)
    assert (tr_human_data.shape == (n_to_keep, n_to_keep)), \
        "The transformation did not filter out the tokens correctly."
    return tr_human_data, tr_machine_data


# ARRTIFICIAL OPERATIONS

def override_machine_with_lower_triangular_of_ones(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Override the machine with a lower triangular matrix of ones."""
    assert human_data.shape == machine_data.shape, "The matrices must have " \
        f"the same shape. Human: {human_data.shape}, machine: {machine_data.shape}"
    assert (human_data.shape[0] == human_data.shape[1]
            and machine_data.shape[0] == machine_data.shape[1]), "The matrices " \
        "must be square."
    # replace all the upper triangle with 0
    n_tokens = machine_data.shape[0]
    transformed_machine_data = np.tril(np.ones((n_tokens, n_tokens)))
    return human_data, transformed_machine_data


def read_probability_and_fill(
        path_csv_probabilities: str,
        col_name_probability: str = "probability",
        col_name_distance: str = "distance",
        pad_length: int = 3000) -> np.ndarray:
    """Read the probabilities and fill the missing distance values.

    It returns:
    - values: the probability values of moving x token away form current token
        note that it is also normalized if the sum is not 1
    - start_position: the position in values of probability from moving 0 token
        away from current token (aka zero distance)

    e.g.
    Input dataframe with these columns and pad_length = 2:
    distance,probability
    -2, 0.2
    0, 0.5
    1, 0.3
    2, 0.2
    4, 0.8

    Output:
    values = [0, 0, 0.1, 0, 0.25, 0.15, 0.1, 0, 0.4, 0, 0]
        # note the addition at position distance=1 and distance=3
    start_position = 4
    """
    df_base = pd.read_csv(path_csv_probabilities)
    # sort based on distance_curent_to_target
    df_base = df_base.sort_values(by=col_name_distance)
    min_position = df_base[col_name_distance].min()
    max_position = df_base[col_name_distance].max()
    safe_values_sum = df_base[col_name_probability].sum()
    positions = np.arange(min_position, max_position + 1)
    values = np.array([
        df_base[df_base[col_name_distance] == p][col_name_probability].values[0]
        if p in df_base[col_name_distance].unique() else 0
        for p in positions
    ])
    # normalize the values
    safe_values_sum = safe_values_sum if safe_values_sum > 0 else 1
    values = values / safe_values_sum
    # ensure that all the positions are present
    assert len(values) == (max_position - min_position + 1), \
        "The base human behavior profile is missing some positions."
    # padd the values with 1000 zeros left and right
    base_values = np.pad(values, (pad_length, pad_length), "constant")
    start_position = - min_position + pad_length
    return base_values, start_position


def remove_base_human_behavior(
        human_data: np.ndarray,
        machine_data: np.ndarray,
        human_metadata: Dict[str, Any],
        machine_metadata: Dict[str, Any],
        tokens_prompt: List[str] = None,
        text_prompt: str = None,
        path_csv_normalization_prob_human: str = None,
        path_csv_normalization_prob_machine: str = None,
        col_name_probability: str = "probability",
        col_name_distance: str = "distance",
        min_base_value: float = 0.01,
        mode: str = 'subtract',
        modelling_based_on_abs_position: bool = False,
        pad_length: int = 3000):
    """Remove the base human behavior ONLY from human ground truth.

    The matrix is expected in the form of a n_token x n_tokens matrix.
    This function scans each matrix row (aka next-token ranking) and converts
    it to a probabilty (sum to 1), and then only the each position is
    normalized by the base human behavior, menaing that the distribution of the
    base human behavior is subtracted (if `mode` == `subtract`), otherwise it is
    divided (if `mode` == `divide`).

    If `modelling_based_on_abs_position` is True, then the base human behavior
    is calculated based on the absolute position of the token in the prompt.
    Thus we expect a larger dataset with columns:
    - `current_token_position`: the position of the current token in the prompt
    - `target_position`: the position of the target token in the prompt
    - `total_tokens`: the total number of tokens in the prompt
    - `p_target`: the probability of the target token being the next token
    Note that this is possible to normalize the human data only.

    """
    if not tokens_prompt:
        tokens_prompt = machine_metadata["tokens_prompt"]
    assert human_data.shape == machine_data.shape, "The matrices must have " \
        f"the same shape. Human: {human_data.shape}, machine: {machine_data.shape}"
    assert (human_data.shape[0] == human_data.shape[1]), "The matrices " \
        f"must be square. Got {human_data.shape}"
    n_next_token_predictions = human_data.shape[0]
    n_possible_next_tokens = human_data.shape[1]

    global df_human_cached
    if path_csv_normalization_prob_human is not None:

        if modelling_based_on_abs_position:
            if df_human_cached is None:
                print('Reading entire dataset from scratch.')
                df_human_cached = pd.read_csv(path_csv_normalization_prob_human)
            df_human = df_human_cached
        else:
            base_values_hum, start_position_hum = read_probability_and_fill(
                path_csv_probabilities=path_csv_normalization_prob_human,
                col_name_probability=col_name_probability,
                col_name_distance=col_name_distance,
                pad_length=pad_length)
    else:
        base_values_hum = np.ones(pad_length * 3)
        start_position_hum = pad_length
    if path_csv_normalization_prob_machine is not None:
        base_values_mach, start_position_mach = read_probability_and_fill(
            path_csv_probabilities=path_csv_normalization_prob_machine,
            col_name_probability=col_name_probability,
            col_name_distance=col_name_distance,
            pad_length=pad_length)
    else:
        base_values_mach = np.ones(pad_length * 3)
        start_position_mach = pad_length

    if modelling_based_on_abs_position:
        closest_n_target_tokens = min([
            original_total_tokens
            for original_total_tokens in df_human["total_tokens"].unique()
            if original_total_tokens >= n_possible_next_tokens
        ])
        # get the probability of the next token
        df_current_vector = df_human[
            (df_human["total_tokens"] == closest_n_target_tokens)
        ]
        # convert to a matrix via the coo format
        normalization_matrix = coo_matrix(
            (df_current_vector["p_target"].values,
             (df_current_vector["current_token_position"].values,
              df_current_vector["target_position"].values)),
            shape=(closest_n_target_tokens, closest_n_target_tokens))
        # convert to a dense matrix
        normalization_matrix = normalization_matrix.todense()
        # restrict to possible next tokens
        normalization_matrix = normalization_matrix[
            :n_possible_next_tokens, :n_possible_next_tokens]

    for i in range(n_next_token_predictions):
        if modelling_based_on_abs_position:
            base_behavior_human = normalization_matrix[i, :]
            # normalize the values
            safe_values_sum = base_behavior_human.sum()
            safe_values_sum = safe_values_sum if safe_values_sum > 0 else 1
            base_behavior_human = base_behavior_human / safe_values_sum
        else:
            # case of constant base human behavior
            # only one vector of probabilities
            base_behavior_human = base_values_hum[
                start_position_hum - i: start_position_hum - i + n_possible_next_tokens]
        # normalize the human data
        safe_sum_human = np.nansum(human_data[i, :])
        safe_sum_human = safe_sum_human if safe_sum_human > 0 else 1
        human_data[i, :] = human_data[i, :] / safe_sum_human
        # normalize the machine data
        safe_sum_machine = np.nansum(machine_data[i, :])
        safe_sum_machine = safe_sum_machine if safe_sum_machine > 0 else 1
        machine_data[i, :] = machine_data[i, :] / safe_sum_machine
        # base behaviour
        base_behavior_machine = base_values_mach[
            start_position_mach - i: start_position_mach - i + n_possible_next_tokens]
        #print('BEFORE')
        #print(f'human_data[{i}, :] = {human_data[i, :]}')
        #print(f'machine_data[{i, :}] = {machine_data[i, :]}')
        if mode == 'subtract':
            human_data[i, :] = human_data[i, :] - base_behavior_human
            machine_data[i, :] = machine_data[i, :] - base_behavior_machine
            #imachine_data[i, :] = machine_data[i, :] - base_behavior
            #print('AFTER').
            #print(f'human_data[{i}, :] = {human_data[i, :]}')
            #print(f'machine_data[{i, :}] = {machine_data[i, :]}')
            #print(f'base_behavior = {base_behavior}')
        elif mode == 'divide':
            # replace the 0s with 1s
            safe_base_behaviour_human = base_behavior_human.copy()
            # add a constant value to avoid division by 0
            safe_base_behaviour_human += min_base_value
            human_data[i, :] = human_data[i, :] / safe_base_behaviour_human
            # replace the 0s with 1s
            safe_base_behaviour_machine = base_behavior_machine.copy()
            safe_base_behaviour_machine += min_base_value
            machine_data[i, :] = machine_data[i, :] / safe_base_behaviour_machine

    return human_data, machine_data

