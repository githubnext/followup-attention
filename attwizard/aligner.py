"""This file processes the output of the models as code."""


import typing
import numpy as np
from typing import List, Dict
from typing import ForwardRef
from typing import List, Dict, Tuple, Any
from pprint import pprint


def distribute_attention(n_chars: int, weight: float, distribution_fn: str):
    """Distribute the attention weight to the tokens.

    Args:
        weight: the attention weight
        distribution_fn: the distribution function
    """
    if distribution_fn == 'equal_share':
        return list(np.ones(n_chars) * (weight / n_chars))
    if distribution_fn == 'replication':
        return list(np.ones(n_chars) * weight)


def replace_char_in_tokens(tokens: List[str], mapping: Dict[str, str]):
    """Replace all characters in the tokens according to the mapping."""
    replaced_tokens = []
    for token in tokens:
        for char_to_replace in mapping.keys():
            token = token.replace(char_to_replace, mapping[char_to_replace])
        replaced_tokens.append(token)
    return replaced_tokens


def map_to_char_level(
        tokens: List[str],
        att_weights: List[float],
        raw_text: str,
        distribution_fn: str = 'replication',
        special_char_mapping: Dict[str, str] = None):
    """Distribute the token attention to individual character.

    distribution_fn: str
        One of the following:
        - 'equal_share' the attention of the word is equally splitted among its
            characters
        - 'replication' all the character of a token receive the token
            attention
    """
    tokens = replace_char_in_tokens(tokens, special_char_mapping)
    pos_in_raw_text = 0
    idx_current_token = 0

    char_weights = []

    while pos_in_raw_text < len(raw_text):
        current_raw_text = raw_text[pos_in_raw_text:]
        current_token = tokens[idx_current_token]
        if current_raw_text.startswith(current_token):
            char_weights += distribute_attention(
                n_chars=len(current_token),
                weight=att_weights[idx_current_token],
                distribution_fn=distribution_fn)
            offset = len(current_token)
            idx_current_token += 1  # go to next token
            pos_in_raw_text += offset  # shift the text pointer
        else:
            pos_in_raw_text += 1
            char_weights.append(0.0)

    return char_weights


def tokenize_char_level(raw_text: str):
    """Tokenize the text in character level.

    Note that we follow the format of the codeattention library so that we can
    easily visualize the heatmap over tokens.
    Example input: "def ...."

    Example of output:
    [
        {'t': 'd', 'i': 0, 'l': 1, 'c': 0},
        {'t': 'e', 'i': 1, 'l': 1, 'c': 1},
        {'t': 'f', 'i': 2, 'l': 1, 'c': 2},
        ...
    ]

    """
    char_tokens = []

    c_line = 1
    c_col = 0

    for idx, char in enumerate(raw_text):
        new_token = {
            't': char,
            'i': idx,
            'l': c_line,
            'c': c_col
        }
        if char == '\n':
            c_line += 1
            c_col = 0
        else:
            c_col += 1
        char_tokens.append(new_token)
    return char_tokens


def get_tokens_with_col_and_line(text, tokens: List[str]):
    """Get tokens with column and line number.

    This function assumes that the concatenation of the tokens matches the text
    as prefix.

    Args:
        text (str): The text to tokenize.
        tokens (list): The tokens to add column and line number to.

    Returns:
        list of dict with the following keys:
            - i: The index of the token in the list of tokens.
            - t: The token text.
            - c: The column number of the token.
            - l: The line number of the token.
            - s: The start index of the token in the text.
    """
    assert text.startswith("".join(tokens))
    tokens_with_col_and_line = []
    line = 0
    column = 0
    tot_prefix_len = 0
    for i, token in enumerate(tokens):
        new_token = {}
        # Check if the token matches the text as prefix.
        if text.startswith(token):
            n_new_lines_char = token.count("\n")

            new_token["s"] = tot_prefix_len
            tot_prefix_len += len(token)
            new_token["i"] = i
            # Add the column and line number to the token.
            new_token["c"] = column
            new_token["l"] = line
            # Update the column and line number.
            column += len(token)
            new_token["t"] = token
            line += n_new_lines_char
            if n_new_lines_char > 0:
                n_chars_after_last_new_line = \
                    len(new_token["t"]) - (new_token["t"].rfind("\n") + 1)
                column = n_chars_after_last_new_line
            # Add the token to the list of tokens with column and line number.
            tokens_with_col_and_line.append(new_token)
            # remove prefix from text
            text = text[len(token):]
        else:
            raise ValueError(
                f"Token {token} does not match text {text}"
            )
    return tokens_with_col_and_line


def convert_weigths_from_tok_to_tok(tokenization, weights, target_tokenization):
    """Convert the weights to the target tokenization.

    This assumes that when the same characters are converted to a single token
    in the new tokenization, the weights are summed.
    On the other hand when one token is converted to multiple tokens in the
    new tokenization, the weight is equally distributed among the new tokens.
    The invariant is that the sum of the weights of the new tokenization is
    equal to the sum of the weights of the old tokenization.
    """
    weights_tok_char = convert_weights_from_tok_to_tok_char(tokenization, weights)
    weights_tok = convert_weights_from_tok_char_to_tok(
        weights_tok_char, target_tokenization
    )
    assert np.isclose(sum(weights), sum(weights_tok))
    return weights_tok


def convert_weights_from_tok_to_tok_char(tokenization, weights):
    """Convert the weights to the char tokenization.

    Note that the weight of a char is derived by the weight of a token divided
    by the number of chars in the token.
    """
    char_weights = []
    for w, t in zip(weights, tokenization):
        new_weight = w / len(t["t"])
        char_weights.extend([new_weight] * len(t["t"]))
    return char_weights


def convert_weights_from_tok_char_to_tok(weights, target_tokenization):
    """Convert the weights to the tokenization.

    Note that the weight of a token is derived by the sum of the weights of the
    chars in the token.
    """
    token_weights = []
    for t in target_tokenization:
        new_weight = sum(weights[t["s"]:t["s"] + len(t["t"])])
        token_weights.append(new_weight)
    return token_weights
