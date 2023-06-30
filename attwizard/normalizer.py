"""This file normalizes the attention tensors."""

import torch
import numpy as np
from typing import List, Dict, Union, Any, ForwardRef, Callable


def norm_convert_row_to_prob(
        att_tensor: Union[np.ndarray, ForwardRef('torch.Tensor')]):
    """Normalize the attention matrix by dividing each row the sum of the row.
    """
    # convert att_tensor to torch.Tensor if necessary
    if not isinstance(att_tensor, torch.Tensor):
        att_tensor = torch.from_numpy(att_tensor)
    # sum the rows of the attention tensor
    sum_row = torch.sum(att_tensor, dim=1).unsqueeze(1)
    matrix_with_row_sums = sum_row.expand(
        size=[-1, att_tensor.shape[1]])
    return att_tensor / matrix_with_row_sums
