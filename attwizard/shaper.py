"""This file manipulates the attention tensor."""

import torch
import numpy as np
from typing import List, Dict, Union, Any, ForwardRef, Callable


def aggregate_dim_tokens_to_line(
        att_tensor: Union[np.ndarray, ForwardRef('torch.Tensor')],
        dim: int,
        agg_fn: Callable = torch.sum,
        line_indices: List[int] = None):
    """Aggregate the given dimension according to the line list.

    Note that the line_indices is a list of line indices, not line numbers.
    They should start from 0 and there should be no gaps. The line indices
    must be in ascending order and the length of the list should match the
    length of the target dimension of the att_tensor.

    Parameters
    ----------
    - att_tensor : the attention tensor to aggregate.
    - dim : the dimension to aggregate.
    - agg_fn : the aggregation function. e.g. use torch.amax or torch.amin.
    - line_indices : the list of line indices to aggregate.
    e.g. a matrix with shape [7 row, 10 columns] and dim 0 to aggregate
    the first dimension, the line_indices could be [0, 0, 0, 1, 2, 2, 2].
    Meaning that the first three rows will be aggregated together, the next
    row will be aggregated together, and the last three rows will be aggregated
    together.
    """
    assert len(line_indices) == att_tensor.shape[dim], \
        "The length of the line_indices must match the " + \
        "length of the target dimension of the att_tensor." \
        "Got {} and {}".format(len(line_indices), att_tensor.shape[dim])
    # assert max(np.diff(line_indices)) == 1, \
    #     "The line indices must be in ascending order and there must be no gaps." + \
    #     "Got {}".format(line_indices)
    # convert att_tensor to torch.Tensor if necessary
    if not isinstance(att_tensor, torch.Tensor):
        att_tensor = torch.from_numpy(att_tensor)
    if not isinstance(line_indices, torch.Tensor):
        line_indices = torch.from_numpy(line_indices)
    sub_tensors = []
    for line_index in list(sorted(line_indices.unique().tolist())):
        idx_target_slices = torch.where(line_indices == line_index)[0]
        slice_tensor = torch.index_select(
            att_tensor, index=idx_target_slices, dim=dim)
        condensed_slice = agg_fn(slice_tensor, dim=dim)
        sub_tensors.append(condensed_slice)

    full_tensor = torch.stack(sub_tensors, dim=dim)
    return full_tensor
