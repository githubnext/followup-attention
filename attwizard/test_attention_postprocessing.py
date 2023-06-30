import pytest
import os
import numpy as np
import torch
import time

from attention_postprocessing import get_follow_up_attention_matrix_v2
from attention_postprocessing import compute_followup_attention

class TestAttentionPostprocessing:

    def test_followup_equivalence(self):
        # read raw attention matrix
        attention_tensor = np.load(os.path.join(
            'test_assets', 'raw_attention.npy'))
        # compute follow-up attention and take the time
        start_time = time.time()
        followup_attention = get_follow_up_attention_matrix_v2(
            attention_tensor=attention_tensor)
        end_time = time.time()
        diff_traditional = end_time - start_time
        # convert to torch tensor
        followup_attention = torch.from_numpy(followup_attention)
        # compute efficient follow-up attention
        start_time = time.time()
        efficient_followup_attention = compute_followup_attention(
            att_tensor=attention_tensor)
        end_time = time.time()
        diff_optimized = end_time - start_time
        # compute the speed-up
        speed_up = diff_traditional / diff_optimized
        print(f"Speed-up: {speed_up:.2f}x")
        # compare the two
        assert np.allclose(
            followup_attention, efficient_followup_attention)