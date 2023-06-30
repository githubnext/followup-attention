"""This file post-process the attention weights to get explanations."""


import numpy as np
from tqdm import tqdm
import networkx as nx

from typing import ForwardRef, List, Tuple, Any, Dict, Union
import torch
import time
from scipy import ndimage

from transformers import AutoTokenizer

from attwizard.decoder import condense_to_4_dim_torch
from attwizard.decoder import visualize_layer_wise_attention
from attwizard.decoder import get_adjacency_matrix
from attwizard.decoder import condense_attention
from attwizard.script.utils import get_model_folder_path


def get_input_and_output_nodes(
        mapping_node_label_to_token_pos: Dict[str, int],
        n_tokens: int,
        n_layers: int = None,):
    """Compute input and output nodes based on the labels."""
    output_nodes = []
    input_nodes = []
    for key in mapping_node_label_to_token_pos.keys():
        if f'L{n_layers}' in key:
            output_nodes.append(key)
        if mapping_node_label_to_token_pos[key] < n_tokens:
            input_nodes.append(key)
    return input_nodes, output_nodes


def compute_attention_flow(
        G: nx.Graph,
        mapping_node_label_to_token_pos: Dict[str, int],
        n_tokens: int,
        n_layers: int = None,):
    """Compute the attention flow from raw attention weights.

    Return: adjacency matrix of the attention flow weights.
    From: “Quantifying Attention Flow In Transformers”, ACL 2020
    """
    input_nodes, _ = get_input_and_output_nodes(
        mapping_node_label_to_token_pos, n_tokens, n_layers)

    number_of_nodes = len(mapping_node_label_to_token_pos)
    flow_values = np.zeros((number_of_nodes, number_of_nodes))
    for key in mapping_node_label_to_token_pos:
        if key not in input_nodes:
            current_layer = int(
                mapping_node_label_to_token_pos[key] / n_tokens)
            pre_layer = current_layer - 1
            u = mapping_node_label_to_token_pos[key]
            for inp_node_key in input_nodes:
                v = mapping_node_label_to_token_pos[inp_node_key]
                flow_value = nx.maximum_flow_value(
                    G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer * n_tokens + v] = flow_value
            flow_values[u] /= flow_values[u].sum()
    adj_matrix_flow_values = flow_values
    return adj_matrix_flow_values


def compute_attention_flow_from_tensor(
        att_tensor: torch.Tensor,
        n_bins_for_weights: int = 100):
    """Compute the attention flow.

    From: “Quantifying Attention Flow In Transformers”, ACL 2020
    """
    start_time = time.time()
    att_tensor = condense_to_4_dim_torch(att_tensor)
    # condense the head (dim 1)
    layerwise_att_matrix = torch.mean(att_tensor, dim=[1])
    # drop condensed dimension
    layerwise_att_matrix = layerwise_att_matrix.squeeze()
    n_layers = layerwise_att_matrix.shape[0]
    # create the graph
    n_tokens = layerwise_att_matrix.shape[1]
    adj_matrix, map_labels_to_token_pos = get_adjacency_matrix(
        att_tensor_layer_by_layer=layerwise_att_matrix,
        input_tokens=list([f"tok{i}" for i in range(n_tokens)])
    )
    G_raw_weights, _ = visualize_layer_wise_attention(
        adj_mat=adj_matrix,
        mapping_node_label_to_token_pos=map_labels_to_token_pos,
        n_layers=n_layers,
        n_tokens=n_tokens,
        n_bin_weights=n_bins_for_weights,
        no_output=True,
    )
    adj_matrix_flow_values = compute_attention_flow(
        G=G_raw_weights,
        mapping_node_label_to_token_pos=map_labels_to_token_pos,
        n_tokens=n_tokens,
        n_layers=n_layers,
    )
    end_time = time.time()
    time_in_seconds = end_time - start_time
    print(f'Time to compute the attention flow: {time_in_seconds} sec.')
    assert adj_matrix_flow_values.shape == (n_layers, n_tokens, n_tokens), \
        f'{adj_matrix_flow_values.shape} != {(n_layers, n_tokens, n_tokens)}' \
        f"the output matrix should have shape (n_layers, n_tokens, n_tokens)"
    raise Exception("Test stop")
    return adj_matrix_flow_values


def compute_attention_rollout(
        layer_wise_att_mat: Union[np.ndarray, ForwardRef('torch.Tensor')],
        add_residual=True):
    """Compute attention rollout from raw attention weights.

    Return: adjacency matrix of the attention rollout weights.
    From: “Quantifying Attention Flow In Transformers”, ACL 2020
    """
    if add_residual:
        residual_att = np.eye(layer_wise_att_mat.shape[1])[None, ...]
        aug_att_mat = layer_wise_att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = layer_wise_att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    # if matrix is not numpy tensor convert it to numpy tensor
    if not isinstance(aug_att_mat, np.ndarray):
        aug_att_mat = aug_att_mat.detach().numpy()
    if not isinstance(joint_attentions, np.ndarray):
        joint_attentions = joint_attentions.detach().numpy()

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])

    return joint_attentions


def compute_attention_rollout_from_tensor(
        att_tensor: torch.Tensor):
    """Compute attention rollout from the attention tensor."""
    att_tensor = condense_to_4_dim_torch(att_tensor)
    # condense the head (dim 1)
    layerwise_att_matrix = torch.mean(att_tensor, dim=[1])
    layerwise_att_matrix_rollout = compute_attention_rollout(
        layer_wise_att_mat=layerwise_att_matrix,
        add_residual=False
    )
    return layerwise_att_matrix_rollout


def sum_over_1st_dimension(
        att_tensor: torch.Tensor,):
    """Sum over the first dimension of the attention tensor."""
    # convert to numpy if not
    if not isinstance(att_tensor, np.ndarray):
        att_tensor = att_tensor.numpy()
    # sum over the first dimension
    condensed_att_tensor = att_tensor.sum(axis=0, keepdims=False)
    return condensed_att_tensor


def raw_weights_last_layer(
        att_tensor: torch.Tensor,):
    """Compute the raw weights of the last layer (condense heads)."""
    att_tensor = condense_to_4_dim_torch(att_tensor)
    # convert to numpy if not
    if not isinstance(att_tensor, np.ndarray):
        att_tensor = att_tensor.numpy()
    # sum over the head dimension
    raw_weights_all_layers = att_tensor.sum(axis=1, keepdims=False)
    return raw_weights_all_layers[-1]


def raw_weights_first_layer(
        att_tensor: torch.Tensor,):
    """Compute the raw weights of the first layer (condense heads)."""
    att_tensor = condense_to_4_dim_torch(att_tensor)
    # convert to numpy if not
    if not isinstance(att_tensor, np.ndarray):
        att_tensor = att_tensor.numpy()
    # sum over the head dimension
    raw_weights_all_layers = att_tensor.sum(axis=1, keepdims=False)
    return raw_weights_all_layers[0]


# REDUCE THE DIMENSION OF THE ATTENTION TENSOR


def max_k_predictions(
        att_tensor: torch.Tensor,
        machine_metadata: List[Dict[str, Any]],
        k: int = 10,):
    """Cut off the attention matrix to max k newly generated tokens.

    This effectively drops the attention referring to the new tokens.
    """
    att_tensor = condense_to_4_dim_torch(att_tensor)
    n_prompt_tokens = len(machine_metadata['tokens_prompt'])
    max_tokens = n_prompt_tokens + k
    # cut off the attention matrix to max k newly generated tokens
    # if it is longer
    assert att_tensor.shape[-1] == att_tensor.shape[-2], \
        f'Attention matrix should be square in the last two dimensions, ' + \
        f'but is {att_tensor.shape}'
    if att_tensor.shape[-1] > max_tokens:
        att_tensor = att_tensor[..., :max_tokens, :max_tokens]
    return att_tensor


def max_10_predictions(
        att_tensor: torch.Tensor,
        machine_metadata: List[Dict[str, Any]]):
    """Cut off the attention matrix to max 10 newly generated tokens.

    This effectively drops the attention referring to the new tokens.
    """
    return max_k_predictions(att_tensor, machine_metadata, k=10)


def max_50_predictions(
        att_tensor: torch.Tensor,
        machine_metadata: List[Dict[str, Any]]):
    """Cut off the attention matrix to max 50 newly generated tokens.

    This effectively drops the attention referring to the new tokens.
    """
    return max_k_predictions(att_tensor, machine_metadata, k=50)


# ABLATION STUDY FOLLOWUP


def extract_single_slice(
        att_tensor: torch.Tensor,
        dim_to_extract: int,
        slice_to_extract: int):
    """Extract a slice over the given dimension."""
    assert dim_to_extract <= 3, \
        f'Currently only dimensions 0, 1, 2, 3 are supported, ' + \
        f'but {dim_to_extract} was given.'
    # convert to numpy if not already
    if not isinstance(att_tensor, np.ndarray):
        att_tensor = att_tensor.numpy()
    if dim_to_extract == 0:
        att_tensor = att_tensor[slice_to_extract, ...]
    elif dim_to_extract == 1:
        att_tensor = att_tensor[:, slice_to_extract, ...]
    elif dim_to_extract == 2:
        att_tensor = att_tensor[:, :, slice_to_extract, ...]
    elif dim_to_extract == 3:
        att_tensor = att_tensor[:, :, :, slice_to_extract, ...]
    return att_tensor


def extract_half_and_sum(
        att_tensor: torch.Tensor,
        dim_to_extract: int,
        half_to_extract: str):
    """Extract half of the attention matrix and sum it up."""
    assert half_to_extract in ["lower", "upper"], \
        f'Currently only "close_to_output" and "close_to_input" are supported, ' + \
        f'but {half_to_extract} was given.'
    assert dim_to_extract <= 3, \
        f'Currently only dimensions 0, 1, 2, 3 are supported, ' + \
        f'but {dim_to_extract} was given.'
    if not isinstance(att_tensor, np.ndarray):
        att_tensor = att_tensor.numpy()
    if half_to_extract == "lower":
        if dim_to_extract == 0:
            return att_tensor[:int(att_tensor.shape[0]/2), ...].sum(axis=0)
        elif dim_to_extract == 1:
            return att_tensor[:, :int(att_tensor.shape[1]/2), ...].sum(axis=1)
        elif dim_to_extract == 2:
            return att_tensor[:, :, :int(att_tensor.shape[2]/2), ...].sum(axis=2)
        elif dim_to_extract == 3:
            return att_tensor[:, :, :, :int(att_tensor.shape[3]/2), ...].sum(axis=3)
    elif half_to_extract == "upper":
        if dim_to_extract == 0:
            return att_tensor[int(att_tensor.shape[0]/2):, ...].sum(axis=0)
        elif dim_to_extract == 1:
            return att_tensor[:, int(att_tensor.shape[1]/2):, ...].sum(axis=1)
        elif dim_to_extract == 2:
            return att_tensor[:, :, int(att_tensor.shape[2]/2):, ...].sum(axis=2)
        elif dim_to_extract == 3:
            return att_tensor[:, :, :, int(att_tensor.shape[3]/2):, ...].sum(axis=3)



# EFFICIENT FOLLOW-UP ATTENTION


def expand_and_normalized(a: torch.Tensor):
    """Expand the N_tokens x N_tokens matrix for the number of tokens.

    Expand a matrix adding a new dimension N_tokens and remove
    - x lines from the top and
    - x columns from the bottom,
    so to leave only a rectangular non-zero section.

    Then normalize these rectangular section in a column wise fashion.
    """
    n_tokens = a.shape[0]
    # mask on rows
    slicer_mat_rows = torch.triu(torch.ones(n_tokens, n_tokens))
    slicer_mat_rows = slicer_mat_rows.unsqueeze(2)
    slicer_mat_rows = slicer_mat_rows.expand(-1, -1, n_tokens)
    # mask on columns
    slicer_mat_col = torch.transpose(slicer_mat_rows, 2, 1)
    slicer_mat_col = torch.flip(slicer_mat_col, [2])
    slicer_mat_col = torch.flip(slicer_mat_col, [0])
    # replicate existing base matrix
    a_stacked = a.unsqueeze(0)
    a_stacked = a_stacked.expand(n_tokens, -1, -1)
    # keep only rectangular sections
    a_stacked_filtered = a_stacked
    a_stacked_filtered = a_stacked_filtered * slicer_mat_rows
    a_stacked_filtered = a_stacked_filtered * slicer_mat_col
    # normalize each column with euclidean norm
    normalization_coeff = torch.norm(a_stacked_filtered, dim=1)
    normalization_coeff = normalization_coeff.unsqueeze(1)
    normalization_coeff = normalization_coeff.expand(-1, n_tokens, -1)
    # Normalize the each value
    a_stacked_normalized = a_stacked_filtered / normalization_coeff
    # replace nan
    a_stacked_normalized = torch.nan_to_num(a_stacked_normalized, nan=0.0)
    return a_stacked_normalized


def compute_from_a_to_b(
        current_level_stacked: torch.Tensor,
        next_level_stacked: torch.Tensor,):
    """Compute a token -> token matrix for consecutive layers."""
    n_tokens = current_level_stacked.shape[0]
    assert current_level_stacked.shape[0] == next_level_stacked.shape[0], "the two inputs must have the same dimensions"
    # generalized multiplication to compute the dot products in parallel
    # this compares the number of followers of a token pairs in two
    # consecutive layers
    res = torch.einsum(
        'bji,bjk->bik',
        current_level_stacked, next_level_stacked)
    # create mask to remove the extra values in the upper left section
    # this is needed since the complete list of followers of a token
    # are exactly all those tokens that follows in the sequence.
    # Note that his sequence is decided by the token of the pair which comes
    # last in the pair (aka we compare the largest set of followers which had
    # the possibility to follow both tokens in the pairs).
    slicer_mat_rows = torch.tril(torch.ones(n_tokens + 1, n_tokens + 1))[:-1, 1:]
    slicer_mat_rows = slicer_mat_rows.unsqueeze(2)
    slicer_mat_rows = slicer_mat_rows.expand(-1, -1, n_tokens)
    transposed = torch.transpose(slicer_mat_rows, 2, 1)
    mask_keep_only_last_slide = 1 - (slicer_mat_rows * transposed)
    mask_keep_only_last_slide
    res = res * mask_keep_only_last_slide
    # condense the stacked version
    return res.sum(dim=0)


def compute_followup_attention_all_layers(att_tensor: torch.Tensor,):
    """Compute the temporal relationship between toknes.

    This models the temporal relationship between the processing which
    happens at two successive layers in the network.

    If the input has 6-dimension we assume that dims are:
    0 is irrelevant (sum over it!)
    1 is layer
    2 isn’t there
    3 is head
    4 is attended from or to
    5 is attended to or from

    if the input has 4-dimension we assume that dims are:
    0 is layer
    1 is head
    2 is attended from or to
    3 is attended to or from

    This function returns the tensor with the following dimensions:
    0 is pair of consecutive layers (this has dimension n_layers - 1)
    1 is the number of tokens
    2 is the number of tokens
    """
    # if numpy tensor convert it to torch tensor
    if isinstance(att_tensor, np.ndarray):
        att_tensor = torch.from_numpy(att_tensor)
    # condense useless dimensions
    if len(att_tensor.shape) == 6:
        att_tensor = att_tensor.sum(2).sum(0)
    assert len(att_tensor.shape) == 4, "the input must have 4 or 6 dimensions"
    # sum all heads
    att_tensor = att_tensor.sum(1)

    all_layers_results = []

    n_layers = att_tensor.shape[0]
    for i in tqdm(range(n_layers-1)):
        # consider all directly consecutive layer pairs
        c_layer = att_tensor[i]
        n_layer = att_tensor[i + 1]

        current_level_stacked = expand_and_normalized(c_layer)
        next_level_stacked = expand_and_normalized(n_layer)

        res = compute_from_a_to_b(
            current_level_stacked, next_level_stacked)

        all_layers_results.append(res)

    # stack results
    all_layers_results = torch.stack(all_layers_results, dim=0)
    print("Consecutive layer pairs: ", all_layers_results.shape)
    return all_layers_results


def compute_followup_attention(att_tensor: torch.Tensor,):
    """Compute the temporal relationship between toknes.

    This models the temporal relationship between the processing which
    happens at two successive layers in the network.

    If the input has 6-dimension we assume that dims are:
    0 is irrelevant (sum over it!)
    1 is layer
    2 isn’t there
    3 is head
    4 is attended from or to
    5 is attended to or from

    if the input has 4-dimension we assume that dims are:
    0 is layer
    1 is head
    2 is attended from or to
    3 is attended to or from
    """
    all_layers_results = compute_followup_attention_all_layers(att_tensor)
    followup_attention = all_layers_results.sum(0)
    return followup_attention


def compute_followup_attention_scaled(att_tensor: torch.Tensor,):
    """Compute the temporal relationship between toknes.

    In this variant later layers are weighted more heavily.
    This models the temporal relationship between the processing which
    happens at two successive layers in the network.

    If the input has 6-dimension we assume that dims are:
    0 is irrelevant (sum over it!)
    1 is layer
    2 isn’t there
    3 is head
    4 is attended from or to
    5 is attended to or from

    if the input has 4-dimension we assume that dims are:
    0 is layer
    1 is head
    2 is attended from or to
    3 is attended to or from
    """
    all_layers_results = compute_followup_attention_all_layers(att_tensor)
    scale_weights = np.arange(1, all_layers_results.shape[0] + 1)
    scale_weights = np.repeat(
        scale_weights[:, np.newaxis], all_layers_results.shape[1], axis=1)
    scale_weights = np.repeat(
        scale_weights[:, :, np.newaxis], all_layers_results.shape[2], axis=2)
    scaled_all_layers_results = all_layers_results * scale_weights
    followup_attention = scaled_all_layers_results.sum(0)
    return followup_attention


def get_follow_up_attention_matrix_v2(
        attention_tensor: Union[np.ndarray, ForwardRef('torch.Tensor')]):
    """
    Extract the follow-up attention for a layer and head.
    Assume that dims are:
    0 is irrelevant (sum over it!)
    1 is layer
    2 isn’t there
    3 is head
    4 is attended from or to
    5 is attended to or from
    """
    assert len(attention_tensor.shape) == 6
    n_layers = attention_tensor.shape[1]
    n_heads = attention_tensor.shape[3]
    n_tokens = attention_tensor.shape[4]
    assert n_tokens == attention_tensor.shape[5], "to and from dimension mismatch"
    attention_tensor = attention_tensor.sum(2).sum(0)
    output = np.zeros((n_tokens, n_tokens, n_layers - 1))
    for k_layer in range(n_layers - 1):
        for i_first_attended in range(n_tokens):
            for j_second_attended in range(n_tokens):
                # only tokens after i and j can be compared:
                max_i_j = max(i_first_attended, j_second_attended)
                # who attended to it in layer k?
                attend_to_i_in_layer = attention_tensor[k_layer, :, max_i_j:, i_first_attended].sum(0)
                # who attended to it in layer k+1?
                attend_to_j_in_next_layer = attention_tensor[k_layer + 1, :, max_i_j:, j_second_attended].sum(0)
                # normalize
                attend_to_i_in_layer = attend_to_i_in_layer / np.linalg.norm(attend_to_i_in_layer)
                attend_to_j_in_next_layer = attend_to_j_in_next_layer / np.linalg.norm(attend_to_j_in_next_layer)
                # take the dot product of the two
                dotproduct = np.dot(attend_to_i_in_layer, attend_to_j_in_next_layer)
                output[i_first_attended, j_second_attended, k_layer] = \
                    0 if np.isnan(dotproduct) else dotproduct
    # sum over heads
    output = output.sum(2)
    return output


# CONDENSE WITH THE MAX

def compute_naive_max_aggregation(
        att_tensor: torch.Tensor,):
    """Compute the max over heads and over layers."""
    att_tensor = condense_to_4_dim_torch(att_tensor)
    # max over all heads and over all layers
    att_tensor = torch.amax(att_tensor, dim=[0, 1])
    return att_tensor


# CONDENSE WITH THE MEAN

def compute_naive_mean_aggregation(
        att_tensor: torch.Tensor,):
    """Compute the mean over heads and over layers."""
    att_tensor = condense_to_4_dim_torch(att_tensor)
    # max over all heads and over all layers
    att_tensor = torch.mean(att_tensor, dim=[0, 1])
    return att_tensor


# CONDENSE A MATRIX WITH THE MEAN OF FOLLOWERS

def compute_mean_of_followers(
        att_tensor: torch.Tensor):
    """Compute the mean of each column (only the lower triang matrix).

    Note that this expects a NxN input with 2-dim only tensors.
    """
    assert len(att_tensor.shape) == 2, "the input must have 2 dimensions"
    # convert to torch tensor if necessary
    if isinstance(att_tensor, np.ndarray):
        att_tensor = torch.from_numpy(att_tensor)
    col_sum = torch.sum(att_tensor, dim=0)
    demominators = torch.arange(len(col_sum), 0, step=-1)
    att_vector = col_sum / demominators
    return att_vector


# TRANSTIVE ATTENTION

def compute_transitive_attention(
        att_tensor: torch.Tensor,
        condense_head_strategy: str = "sum",
        alpha: float = 0.5,
        beta: float = 0.5,
        mult_direction: str = "left"):
    """Compute the transitive attention.

    The transitive attention at leyer l is computed as follows:
    Tra(l) = Att(l) x ((1 - alpha) * I_n + alpha * Tra(l-1)) + beta * Tra(l-1)

    Att(l) are the raw attention weights at layer l.
    The symbol X stands for matrix multiplication.
    Note that the above formula is for the left multiplication case; whereas
    the right multiplication case is:
    Tra(l) = ((1 - alpha) * I_n + alpha * Tra(l-1)) X Att(l) + beta * Tra(l-1)

    """
    assert mult_direction in ["left", "right"], \
        "mult_direction must be left or right"
    print("Alpha: ", alpha)
    print("Beta: ", beta)
    print("Condense head strategy: ", condense_head_strategy)
    print("Mult direction: ", mult_direction)
    att_tensor = condense_to_4_dim_torch(att_tensor)
    layer_wise_attention = condense_attention(
        attention_tensor=att_tensor,
        reduce_direction='head',
        reduce_function=condense_head_strategy)
    layer_wise_attention = torch.squeeze(layer_wise_attention)

    # compute the transitive attention at each layer
    transitive_attention = torch.clone(layer_wise_attention)
    for k_layer in range(1, layer_wise_attention.shape[0]):
        # compute the transitive attention at layer k
        att_l = layer_wise_attention[k_layer]
        tra_l_prev = transitive_attention[k_layer - 1]
        if mult_direction == "left":
            tra_l = torch.matmul(
                att_l,
                (1 - alpha) * torch.eye(att_l.shape[0]) + alpha * tra_l_prev
                ) + beta * tra_l_prev
        elif mult_direction == "right":
            tra_l = torch.matmul(
                (1 - alpha) * torch.eye(att_l.shape[0]) + alpha * tra_l_prev,
                att_l
                ) + beta * tra_l_prev
        transitive_attention[k_layer] = tra_l
    return transitive_attention


# OTHERS

def make_symmetric(
        att_tensor: torch.Tensor,
        strategy: str = "sum"):
    """Make the attention matrix symmetric."""
    assert strategy in ["max", "mean", 'sum'], \
        "strategy must be max or mean or sum"
    # check that the dim is two
    assert len(att_tensor.shape) == 2, "the input must have 2 dimensions"
    # convert to numpy if necessary
    if isinstance(att_tensor, torch.Tensor):
        att_tensor = att_tensor.numpy()
    # make the matrix symmetric
    if strategy == "max":
        att_tensor = np.maximum(att_tensor, att_tensor.T)
    elif strategy == "mean":
        att_tensor = (att_tensor + att_tensor.T) / 2
    elif strategy == "sum":
        att_tensor = att_tensor + att_tensor.T
        # divide the diagonal by 2
        np.fill_diagonal(att_tensor, att_tensor.diagonal() / 2)
    return att_tensor


def generate_uniform_attention(
        att_tensor: torch.Tensor):
    """Generate a uniform matrix with the same shape as the attention tensor."""
    n_tokens = att_tensor.shape[-1]
    matrix = np.tril(np.ones((n_tokens, n_tokens)))
    # normalize matrix by line
    summation_vector = matrix.sum(axis=1, keepdims=True)
    # replace nan with 1
    summation_vector = np.nan_to_num(summation_vector, nan=1)
    # replace 0 with 1
    summation_vector[summation_vector == 0] = 1
    # divide each line by the sum
    matrix = matrix / summation_vector
    matrix = np.nan_to_num(matrix, nan=0)
    return matrix


def generate_gaussian_attention_in_neighborhood_from_metadata(
        metadata: Dict[str, Any],
        sigma: float = 10):
    """Generate a matrix in which each row gives attention to its neighbords.

    The attention is distributed as a gaussian around the current line.
    """
    raw_text = metadata["text_prompt"]
    model_name = metadata["mode_name"]
    config = metadata["config_options"]

    # get the vocab
    model_folder_path = get_model_folder_path(
        model_folder=config["local_model_folder"],
        hugging_short_repo=model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

    input_ids = tokenizer(raw_text, return_tensors="pt")['input_ids'][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    n_tokens = len(tokens)
    m = np.diag(np.ones(n_tokens))
    for i in range(n_tokens):
        row = m[i, :]
        # convert to float
        row = row.astype(float)
        m[i, :] = ndimage.gaussian_filter1d(
            input=np.float_(row),
            sigma=sigma,
            cval=0.0,
            mode='constant',)
    return m


def generate_uniform_attention_from_metadata(
        metadata: Dict[str, any]):
    """Generate a uniform matrix with the same shape as the number of tokens."""
    raw_text = metadata["text_prompt"]
    model_name = metadata["mode_name"]
    config = metadata["config_options"]

    # get the vocab
    model_folder_path = get_model_folder_path(
        model_folder=config["local_model_folder"],
        hugging_short_repo=model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

    input_ids = tokenizer(raw_text, return_tensors="pt")['input_ids'][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    n_tokens = len(tokens)
    matrix = np.tril(np.ones((n_tokens, n_tokens)))
    # normalize matrix by line
    summation_vector = matrix.sum(axis=1, keepdims=True)
    # replace nan with 1
    summation_vector = np.nan_to_num(summation_vector, nan=1)
    # replace 0 with 1
    summation_vector[summation_vector == 0] = 1
    # divide each line by the sum
    matrix = matrix / summation_vector
    matrix = np.nan_to_num(matrix, nan=0)
    return matrix


def generate_copy_cat_attention(
        metadata: Dict[str, any]):
    """Create a matrix that predicts the position containing the same token."""
    raw_text = metadata["text_prompt"]
    model_name = metadata["mode_name"]
    config = metadata["config_options"]

    # get the vocab
    model_folder_path = get_model_folder_path(
        model_folder=config["local_model_folder"],
        hugging_short_repo=model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

    input_ids = tokenizer(raw_text, return_tensors="pt")['input_ids'][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    size = len(tokens)
    copy_cat_att = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if tokens[i] == tokens[j]:
                copy_cat_att[i][j] = 1
    return copy_cat_att