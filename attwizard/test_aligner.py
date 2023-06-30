import os
import pytest

from aligner import map_to_char_level
from aligner import tokenize_char_level
from aligner import convert_weigths_from_tok_to_tok
from aligner import get_tokens_with_col_and_line
from pprint import pprint


@pytest.fixture()
def prompt():
    yield """def sum(a, b):
    c ="""


@pytest.fixture()
def language_model_tokens():
    yield [
        'def', 'Ġsum',  '(', 'a', ',', 'Ġb',
        '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġc', 'Ġ=']


@pytest.fixture()
def language_model_attention_weights():
    yield [
        3, 3,  1, 1, 1, 2,
        2, 1, 1, 1, 1, 2, 2]


@pytest.fixture()
def py_lexer_tokens():
    yield [
        'def', 'sum',  '(', 'a', ',', 'b',
        ')', ':', 'c', '=']


@pytest.fixture()
def py_lexer_weights():
    yield [
        3, 3, 1, 1, 1, 1,
        1, 1, 1, 1]


@pytest.fixture()
def special_char_map():
    yield {
        "Ġ": ' ',
        "Ċ": '\n',
    }


@pytest.fixture()
def tok_1():
    yield ["a", "b ", "d", "e", " ", "gh", ".", "\n", " ", "a", " ", "bc", "."]

@pytest.fixture()
def tok_2():
    yield ["ab de", " ", "gh.\n", " ", "a bc."]


@pytest.fixture()
def prompt_random():
    yield """ab de gh.\n a bc."""



class TestAligner:

    def test_conv_to_char_level_w_replication(
            self,
            prompt,
            language_model_tokens,
            language_model_attention_weights,
            special_char_map):
        """Test the conversion to char level of a token-level attention."""
        char_weights = map_to_char_level(
            tokens=language_model_tokens,
            att_weights=language_model_attention_weights,
            raw_text=prompt,
            distribution_fn='replication',
            special_char_mapping=special_char_map)
        assert char_weights == [
            3, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0,
            2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

    def test_conv_to_char_level_w_equal_share(
            self,
            prompt,
            language_model_tokens,
            language_model_attention_weights,
            special_char_map):
        """Test the conversion to char level of a token-level attention."""
        char_weights = map_to_char_level(
            tokens=language_model_tokens,
            att_weights=language_model_attention_weights,
            raw_text=prompt,
            distribution_fn='equal_share',
            special_char_mapping=special_char_map)
        assert char_weights == [
            1, 1, 1, .75, .75, .75, .75, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_conv_to_char_level_w_replication_on_code(
            self,
            prompt,
            py_lexer_tokens,
            py_lexer_weights,
            special_char_map):
        """Test the conversion to char level of a token-level att on code."""
        char_weights = map_to_char_level(
            tokens=py_lexer_tokens,
            att_weights=py_lexer_weights,
            raw_text=prompt,
            distribution_fn='replication',
            special_char_mapping=special_char_map)
        assert char_weights == [
            3, 3, 3, 0, 3, 3, 3, 1, 1, 1, 0, 1, 1, 1, 0,
            0, 0, 0, 0, 1, 0, 1]

    def test_tokenize_char_level(
            self,
            prompt):
        tokens = tokenize_char_level(prompt)
        assert tokens == [
            {'t': 'd', 'i': 0, 'l': 1, 'c': 0},
            {'t': 'e', 'i': 1, 'l': 1, 'c': 1},
            {'t': 'f', 'i': 2, 'l': 1, 'c': 2},
            {'t': ' ', 'i': 3, 'l': 1, 'c': 3},
            {'t': 's', 'i': 4, 'l': 1, 'c': 4},
            {'t': 'u', 'i': 5, 'l': 1, 'c': 5},
            {'t': 'm', 'i': 6, 'l': 1, 'c': 6},
            {'t': '(', 'i': 7, 'l': 1, 'c': 7},
            {'t': 'a', 'i': 8, 'l': 1, 'c': 8},
            {'t': ',', 'i': 9, 'l': 1, 'c': 9},
            {'t': ' ', 'i': 10, 'l': 1, 'c': 10},
            {'t': 'b', 'i': 11, 'l': 1, 'c': 11},
            {'t': ')', 'i': 12, 'l': 1, 'c': 12},
            {'t': ':', 'i': 13, 'l': 1, 'c': 13},
            {'t': '\n', 'i': 14, 'l': 1, 'c': 14},
            {'t': ' ', 'i': 15, 'l': 2, 'c': 0},
            {'t': ' ', 'i': 16, 'l': 2, 'c': 1},
            {'t': ' ', 'i': 17, 'l': 2, 'c': 2},
            {'t': ' ', 'i': 18, 'l': 2, 'c': 3},
            {'t': 'c', 'i': 19, 'l': 2, 'c': 4},
            {'t': ' ', 'i': 20, 'l': 2, 'c': 5},
            {'t': '=', 'i': 21, 'l': 2, 'c': 6}
        ]

    def test_conversion_of_tokenization_systems(
            self,
            prompt_random,
            tok_1,
            tok_2):
        """Test the conversion of tokenization systems."""
        weights_2 = [1, 1, 1, 1, 1]
        tokenization_2 = get_tokens_with_col_and_line(prompt_random, tok_2)
        pprint(tokenization_2)

        pprint("Target tokenization 1")
        tokenization_1 = get_tokens_with_col_and_line(prompt_random, tok_1)
        pprint(tokenization_1)
        weights_1 = convert_weigths_from_tok_to_tok(
            tokenization_2, weights_2, tokenization_1)
        pprint(weights_1)

    def test_conversion_of_tokenization_system_with_double_new_lines(
            self):
        """Test the conversion of tokenization systems."""
        prompt = """ab de gh.\n\n a \nbc."""
        tok_1 = ["ab", " ", "de", " ", "gh.\n\n", " ", "a", " ", "\nbc."]
        tok_2 = ["ab de", " ", "gh.\n\n", " ", "a \nbc."]
        weights_2 = [1, 1, 1, 1, 1]
        tokenization_2 = get_tokens_with_col_and_line(prompt, tok_2)
        pprint(tokenization_2)

        pprint("Target tokenization 1")
        tokenization_1 = get_tokens_with_col_and_line(prompt, tok_1)
        pprint(tokenization_1)
        weights_1 = convert_weigths_from_tok_to_tok(
            tokenization_2, weights_2, tokenization_1)
        pprint(weights_1)
