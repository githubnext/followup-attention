"""This file processes and visualizes the masked attention of decoder."""

import os

import pandas as pd
import numpy as np
import typing
from typing import Callable
from typing import ForwardRef, List, Tuple, Any, Dict

import transformers
import torch
import torch.nn.functional as F


from attwizard.visualizer.matrix import heatmap_visualize
from attwizard.visualizer.matrix import visualize_layer_wise_attention

# MANIPULATION OF ATTENTION TENSORS


def get_attention_tensor(
        model_output: transformers.generation_utils.SampleDecoderOnlyOutput):
    """Extract a tensor representing the entire attention.

    Note that the output will be 6-dimensional:
    e.g. (6predictions, 12layers, 1, 12heads, 9tokens, 9tokens)
    """
    attentions = model_output["attentions"]

    last_new_token_attention = attentions[-1]
    n_layers = len(last_new_token_attention)
    last_new_token_attention_1st_layer = last_new_token_attention[0]
    # last_new_token_attention_1st_layer.shape [1, 16, 1, 19]
    n_heads = last_new_token_attention_1st_layer.shape[1]
    n_tokens = last_new_token_attention_1st_layer.shape[-1]

    # pad all the tensors with zeros

    all_atts_tensor_list = []
    # for each generated token i
    for i in range(len(attentions)):

        all_atts_tensor_list_for_i_pred = []
        # for each layer j
        for j in range(n_layers):

            padded_attention = attentions[i][j].detach().cpu()

            # pad last dimension
            i_dimension = attentions[i][j].shape[-1]
            pad_to_add = n_tokens - i_dimension
            padded_attention = F.pad(
                padded_attention, pad=(0, pad_to_add),
                mode='constant', value=0)

            # pad second-last dimension
            second_last_i_dimension = attentions[i][j].shape[-2]
            pad_before = i_dimension - second_last_i_dimension
            pad_after = n_tokens - pad_before - second_last_i_dimension
            padded_attention = F.pad(
                padded_attention, pad=(0, 0, pad_before, pad_after),
                mode='constant', value=0)

            all_atts_tensor_list_for_i_pred.append(padded_attention)
            # assert attentions[0][j].shape == (1, 12, 4, 4)
            # assert attentions[1][j].shape == (1, 12, 1, 5)
            # assert attentions[2][j].shape == (1, 12, 1, 6)
            # assert attentions[3][j].shape == (1, 12, 1, 7)
            # assert attentions[4][j].shape == (1, 12, 1, 8)
            # assert attentions[5][j].shape == (1, 12, 1, 9)
        att_all_layers_i_predictions = torch.stack(
            all_atts_tensor_list_for_i_pred
        )

        all_atts_tensor_list.append(att_all_layers_i_predictions)
    att_tensor = torch.stack(all_atts_tensor_list)

    return att_tensor


def condense_attention(
        attention_tensor: typing.Union[np.ndarray, ForwardRef('torch.Tensor')],
        reduce_direction: str,
        reduce_function: typing.Union[Callable, str] = torch.sum):
    """Condense the attention tensor along one direction.

    Parameters:
        reduce_direction: str, one of 'layer', 'head'. If layer-wise all the
            layers are condensed, if head-wise all the heads are condensed.
        attention_tensor: np.ndarray or torch.Tensor, the attention tensor.
        function: str, the function to use to condense the attention, one of
            torch.amin, torch.sum, torch.amax, torch.mean. If none, the same
            attention tensor as input is returned.

    Returns:
        np.ndarray or torch.Tensor, the condensed attention tensor.
    """
    assert reduce_direction in ('layer', 'head', 'both')
    if reduce_function is None:
        return attention_tensor
    # if 6-dimensional input
    # (6predictions, 12layers, 1, 12heads, 9tokens, 9tokens)
    if len(attention_tensor.shape) == 6:
        map_to_dim = {
            'layer': 1,
            'head': 3,
        }
    # if 4-dimensional input
    # (12layers, 12heads, 9tokens, 9tokens)
    if len(attention_tensor.shape) == 4:
        map_to_dim = {
            'layer': 0,
            'head': 1,
        }
    # convert to torch tensor if numpy tensor
    if isinstance(attention_tensor, np.ndarray):
        attention_tensor = torch.from_numpy(attention_tensor)
    if isinstance(reduce_function, str):
        if reduce_function == 'sum':
            reduce_function = torch.sum
        elif reduce_function == 'mean':
            reduce_function = torch.mean
        elif reduce_function == 'max':
            reduce_function = torch.amax
        else:
            raise ValueError(
                "reduce_function must be one of 'sum', 'mean', 'max'")
    if reduce_direction == 'both':
        return reduce_function(
            attention_tensor,
            dim=[map_to_dim['layer'], map_to_dim['head']], keepdim=True)
    else:
        return reduce_function(
            attention_tensor,
            dim=map_to_dim[reduce_direction], keepdim=True)


def condense_to_4_dim_torch(
        att_tensor: typing.Union[np.ndarray, ForwardRef('torch.Tensor')]):
    """Condense the attention tensor to 4 dimensions (output torch type).

    The output has the following dimensions:
    - 0 is layer
    - 1 is head
    - 2 is token which is giving attention
    - 3 is token which is being attended to
    """
    # if numpy tensor convert it to torch tensor
    if isinstance(att_tensor, np.ndarray):
        att_tensor = torch.from_numpy(att_tensor)
    # condense useless dimensions
    if len(att_tensor.shape) == 6:
        att_tensor = att_tensor.sum(2).sum(0)
    assert len(att_tensor.shape) == 4, "the input must have 4 or 6 dimensions"
    return att_tensor


def get_attention_matrix(
        attention_tensor: typing.Union[np.ndarray, ForwardRef('torch.Tensor')],
        layer: int = None, head: int = None):
    """Extract the self-attention for a layer and head."""
    if not layer and type(layer) != int:
        assert attention_tensor.shape[1] == 1, "Layer must be specified"
        layer = 0
    if not head and type(head) != int:
        assert attention_tensor.shape[3] == 1, "Head must be specified"
        head = 0

    return attention_tensor[:, layer, :, head].squeeze()


def merge_attention_prompt_and_new_tokens(
        attention_tensor: typing.Union[np.ndarray, ForwardRef('torch.Tensor')]
        ):
    """Merge self-attention on prompt and the attention on the new tokens.

    The attention on the prompt is computed once, then for each newly generated
    token only its attention is returned.
    This function merges both in a larger tensor. For example, with a prompt of
    30 tokens and a max_length generation of 100, we have a 30x30 tensor, and
    70 tensors of dimensions 1X31, 1X32, ..., 1X100.
    The output of the function gives a tensor of dimensions 100X100, which is
    padded with zeros.

    Note that the attention tensor is already 100x100, this makes sure to sum
    all the vectors corresponding to different new tokens predictions.
    """
    assert len(attention_tensor.shape) == 6, \
        "Attention tensor must be 6-dimensional"
    return torch.sum(attention_tensor, dim=0, keepdim=True)


def get_attention_representation(
        att_matrix: typing.Union[np.ndarray, ForwardRef('torch.Tensor')],
        token_pos: int):
    """Extract the attention representation for a token.

    Parameters:
        att_matrix: np.ndarray or torch.Tensor, the square attention matrix.
        token_pos: int, the position of the token in the generated sequence.
    """
    return att_matrix[token_pos, :]


def get_adjacency_matrix(
        att_tensor_layer_by_layer: typing.Union[
            np.ndarray, ForwardRef('torch.Tensor')],
        input_tokens: List[str]):
    """Compute the adjacency matrix of the layer-by-layer graph.

    The layer-by-layer graph consists of the a graph of:
    sequence_length x (n_layers + 1) nodes.
    The root nodes are the input embeddings and the leaves are the output
    embeddings. In between, there are the hidden embeddings which are connected
    to the previous layers via attention weights.
    Each node has edges which points to nodes in the previous layer and which
    are combined, according to the attention weights, to obtain the current
    node.

    Parameters:
        att_tensor_layer_by_layer: np.ndarray or torch.Tensor, the attention
            weights for each layer. The tensor should have dimension 3.
            n_layers x sequence_length x sequence_length.
        input_tokens: List[str], the input tokens.

    Returns:
        adj_mat: np.ndarray, the adjacency matrix.
        mapping_node_label_to_token_pos: Dict[str, int], the mapping from
            labels of a node to position index of the token in the sequence.

    Credits:
    Inspired by code from: “Quantifying Attention Flow In Transformers”
    """
    if isinstance(att_tensor_layer_by_layer, torch.Tensor):
        att_tensor_layer_by_layer = torch.squeeze(att_tensor_layer_by_layer)
    elif isinstance(att_tensor_layer_by_layer, np.ndarray):
        att_tensor_layer_by_layer = np.squeeze(att_tensor_layer_by_layer)
    assert len(att_tensor_layer_by_layer.shape) == 3
    assert att_tensor_layer_by_layer.shape[1] == len(input_tokens)
    assert att_tensor_layer_by_layer.shape[2] == len(input_tokens)
    # check if the att_tensor_layer_by_layer is pytorch and convert to numpy
    if isinstance(att_tensor_layer_by_layer, torch.Tensor):
        att_tensor_layer_by_layer = att_tensor_layer_by_layer.detach().numpy()
    n_layers, length, _ = att_tensor_layer_by_layer.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    mapping_node_label_to_token_pos = {}
    for k in np.arange(length):
        mapping_node_label_to_token_pos[str(k)+"_"+input_tokens[k]] = k

    for i in np.arange(1, n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            mapping_node_label_to_token_pos[label] = index_from
            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = \
                    att_tensor_layer_by_layer[i-1][k_f][k_t]

    return adj_mat, mapping_node_label_to_token_pos


def normalize_less_attention_on_early_tokens(
        att_tensor: typing.Union[np.ndarray, ForwardRef('torch.Tensor')]):
    """Normalize attention based on the distance from the first token.

    Intuition, the first token will receive attention from all the other
    tokens, thus it is advantaged, over the last token in the prompt,
    because it will be attended by no-one.

    """
    assert len(att_tensor.shape) == 6, \
        "Attention tensor must be 6-dimensional"
    new_att_tensor = att_tensor
    for i in np.arange(att_tensor.shape[4]):
        new_att_tensor[:, :, :, :, i, :] *= (i + 1) ** (0.5)
    return new_att_tensor


def add_residual_weights(
        att_tensor: typing.Union[np.ndarray, ForwardRef('torch.Tensor')]):
    """Add to the attention the contribution for the residual connections.

    The residual connections are added to the raw attention weights, since the
    transformer after computing the self-attention has a skip connection which
    sums the original embedding to the newly obtained weighted sum.

    Note that this function takes only a tensor with the following dimensions
    as input: n_layers x sequence_length x sequence_length
    """
    assert len(att_tensor.shape) == 3, \
        "Attention tensor must be 3-dimensional: " + \
        "n_layers x sequence_length x sequence_length. "
    res_tensor = np.eye(att_tensor.shape[-1])[None, ...]
    att_tensor = 0.5 * att_tensor + 0.5 * res_tensor
    return att_tensor


def get_reduce_fn(strategy: str) -> Callable:
    """Depending on the string return the correct torch function."""
    if strategy == "max":
        return torch.amax
    elif strategy == "mean":
        return torch.mean
    elif strategy == "sum":
        return torch.sum
    elif strategy == "min":
        return torch.amin
    elif strategy == "keep":
        return None
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))


def extract_att_matrix(
        model_output: Dict[str, Any],
        condense_on_head_dim: str,
        condense_on_layer_dim: str,
        normalization_strategy: str):
    """Get the condensed attention matrix for the given model.

    Note that the model output must include the attention information in the
    key "attentions". Typically you have to call the model with the flag
    output_attentions=True, e.g.
    model.generate(..., output_attentions=True, return_dict_in_generate=True)
    """
    att_tensor = get_attention_tensor(model_output=model_output)
    if normalization_strategy == "more_weight_to_recent":
        att_tensor = normalize_less_attention_on_early_tokens(att_tensor)
    condensed_att_tensor = \
        merge_attention_prompt_and_new_tokens(att_tensor)
    condensed_att_tensor = \
        condense_attention(
            condensed_att_tensor,
            reduce_direction="head",
            reduce_function=get_reduce_fn(condense_on_head_dim))
    condensed_att_tensor = \
        condense_attention(
            condensed_att_tensor,
            reduce_direction="layer",
            reduce_function=get_reduce_fn(condense_on_layer_dim))
    att_matrix = condensed_att_tensor.squeeze()
    return att_matrix
