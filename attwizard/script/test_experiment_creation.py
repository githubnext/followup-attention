import os
import pytest
import random

from attwizard.script.experiment_creation import produce_assignment_file_and_attention


@pytest.fixture()
def special_char_map():
    yield """a = 1
b = 3
sum = a + b
if (sum =="""


class TestCreation:

    def test_correctness_relavance(
            self):
        """Test correct statement relevance."""
        # set seed
        random.seed(0)
        content, relevance_metadata = produce_assignment_file_and_attention()
        assert len(relevance_metadata["attention_weights"]) == len(content.split("\n"))