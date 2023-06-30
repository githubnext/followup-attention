import os
import pytest
import torch

import transformers
from transformers import AutoTokenizer
from transformers.generation_utils import SampleDecoderOnlyOutput

from decoder import get_attention_tensor
from decoder import merge_attention_prompt_and_new_tokens
from decoder import condense_attention
from decoder import get_attention_matrix

MODEL_FOLDER = "../huggingface_models/gpt-neo"
REMOTE_REPO = "EleutherAI/gpt-neo-125M"
PROMPT_TEXT = "hello how are you"

TEST_ASSET_FOLDER = "test_assets"


@pytest.fixture()
def tokenizer():
    print("setup: creating a tokenizer")
    if os.path.exists(os.path.join(MODEL_FOLDER, "pytorch_model.bin")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    else:
        tokenizer = AutoTokenizer.from_pretrained(REMOTE_REPO)
        tokenizer.save_pretrained(MODEL_FOLDER)
    yield tokenizer


@pytest.fixture()
def input_ids(tokenizer):
    yield tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids


@pytest.fixture()
def model_output():
    print("setup: creating a prediction")
    torch.manual_seed(42)
    model_output = SampleDecoderOnlyOutput(
        sequences=torch.load(
            os.path.join(TEST_ASSET_FOLDER, "output_sequences_gpt_neo.pt")),
        attentions=torch.load(
            os.path.join(TEST_ASSET_FOLDER, "output_attentions_gpt_neo.pt")),
    )
    yield model_output


class TestTensorManipulation:

    def test_output_of_the_model(
            self, model_output):
        """Test the creation of the attention tensor."""
        att_output = model_output["attentions"]

        # check that only 6 new tokens are produced
        # prompt: 'hello how are you'
        # output: 'hello how are you doing?\n\n<m'
        assert len(att_output) == 6

        # check that the first tensor represents the self-attention on the
        # first four tokens
        for j in range(12):  # for all layers
            assert att_output[0][j].shape == (1, 12, 4, 4)
            assert att_output[1][j].shape == (1, 12, 1, 5)
            assert att_output[2][j].shape == (1, 12, 1, 6)
            assert att_output[3][j].shape == (1, 12, 1, 7)
            assert att_output[4][j].shape == (1, 12, 1, 8)
            assert att_output[5][j].shape == (1, 12, 1, 9)

    def test_creation_of_attention_tensor(
            self, model_output):
        """Test the creation of the attention tensor."""
        att_tensor = get_attention_tensor(
            model_output=model_output
        )
        new_predictions = 6
        assert att_tensor.shape == (new_predictions, 12, 1, 12, 9, 9)

    def test_merging_tensor(self, model_output):
        """Test the condensation of a tensor over all the predicted tokens."""
        att_tensor = get_attention_tensor(
            model_output=model_output
        )
        condensed_att_tensor = \
            merge_attention_prompt_and_new_tokens(att_tensor)
        assert condensed_att_tensor.shape == (1, 12, 1, 12, 9, 9)

    def test_condense_layers(self, model_output):
        """Test the condensation of a tensor over layer direction."""
        att_tensor = get_attention_tensor(
            model_output=model_output
        )
        condensed_att_tensor = \
            condense_attention(
                att_tensor, reduce_direction="layer")
        assert condensed_att_tensor.shape == (6, 1, 1, 12, 9, 9)

    def test_condense_heads(self, model_output):
        """Test the condensation of a tensor over head direction."""
        att_tensor = get_attention_tensor(
            model_output=model_output
        )
        condensed_att_tensor = \
            condense_attention(
                att_tensor, reduce_direction="head")
        assert condensed_att_tensor.shape == (6, 12, 1, 1, 9, 9)

    def test_get_attention_matrix(self, model_output):
        """Get the attention matrix for a specific layer and head."""
        att_tensor = get_attention_tensor(
            model_output=model_output
        )
        condensed_att_tensor = \
            merge_attention_prompt_and_new_tokens(att_tensor)
        att_matrix = get_attention_matrix(
            attention_tensor=condensed_att_tensor, layer=2, head=4)
        assert att_matrix.shape == (9, 9)
