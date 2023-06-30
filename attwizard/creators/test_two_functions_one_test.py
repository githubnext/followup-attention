from ast import Constant
import os
import numpy as np
import pytest
import torch

from attwizard.creators.two_functions_one_test import TwoFunctionsOneTestCreator

@pytest.fixture()
def prompt():
    yield """def sum(a, b):
    c ="""


@pytest.fixture()
def full_function_string():
    yield "".join([
        'def has_close_elements(numbers: List[float], threshold: float) -> bool:',
        '    """ Check if in given list of numbers, are any two numbers closer to '
        'each other than',
        '    given threshold.',
        '    """',
        '    for idx, elem in enumerate(numbers):',
        '        for idx2, elem2 in enumerate(numbers):',
        '            if idx != idx2:',
        '                distance = abs(elem - elem2)',
        '                if distance < threshold:',
        '                    return True',
        '',
        '    return False',
        '',
        '# Write a test for the function has_close_elements below',
        'assert has_close_elements('
    ])


class TestAssignmentChainCreator:

    def test_function_test_separation(self, full_function_string):
        """Test the extraction of a simple function's parts."""
        creator = TwoFunctionsOneTestCreator(
            name="xxx", prompt_folder="xxx", ground_truth_folder="xxx",
        )