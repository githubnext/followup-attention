from ast import Constant
import os
import numpy as np
import pytest
import torch

from attwizard.creators.assignement_chain import AssignmentChainCreator
from attwizard.creators.dep_graph import *
from attwizard.creators.distance_to_weights import convert_distance_to_weight_naive

@pytest.fixture()
def prompt():
    yield """def sum(a, b):
    c ="""


@pytest.fixture()
def kwargs():
    yield {
        "relevant_variable": "a",
        "supporting_variables": ["b", "c"],
        "useless_variables": ["d", "e", "f"],
        "prefix_op": [
            {"name": "sum", "types": ["List[int]"]},
        ],
        "infix_op": [
            {"name": "+", "types": ["int", "int"]},
        ],
        "assignemnt_chain_length": 3,
        "n_useless_operations": 10,
        "custom_prompt": 'print(<RELEVANT_VARIABLE>) # prints the value "'
    }


class TestAssignmentChainCreator:

    def test_simple_graph_creation(self, kwargs):
        """Simple graph with relevant var declaration and prompt statement."""
        prog = Program()
        # a = 3
        rel_node = DepNode(type="int", identifier_name="a")
        const_node = DepNode(type="int", identifier_name="3", literal=True)
        declaration = AssignmentOp(input_nodes=[const_node], output_nodes=[rel_node])
        prog.safe_append_op(declaration)
        # print(a) # prints the value "
        query =  QueryOp(
            query_statement=kwargs["custom_prompt"],
            relevant_variable=rel_node)
        prog.safe_append_op(query)

        prompt_text = prog.get_full_output()

        assert prompt_text == "\n".join([
            "a = 3",
            'print(a) # prints the value "'
        ]) + "\n"

    def test_arithmetic_sum(self, kwargs):
        """Chechk the creation of an arithmetic sum.
        Note that the types must be compatible."""
        prog = Program()
        # a = 3
        a_node = DepNode(type="int", identifier_name="a")
        const_node = DepNode(type="int", identifier_name="3", literal=True)
        declaration = AssignmentOp(input_nodes=[const_node], output_nodes=[a_node])
        prog.safe_append_op(declaration)
        # b = 5
        b_node = DepNode(type="int", identifier_name="b")
        const_node = DepNode(type="int", identifier_name="5", literal=True)
        declaration = AssignmentOp(input_nodes=[const_node], output_nodes=[b_node])
        prog.safe_append_op(declaration)
        # c = a + b
        c_node = DepNode(type="int", identifier_name="c")
        sum_node = InfixOp(
            op_symbol="+",
            input_nodes=[a_node, b_node],
            output_nodes=[c_node])
        prog.safe_append_op(sum_node)

        prompt_text = prog.get_full_output()

        assert prompt_text == "\n".join([
            "a = 3",
            "b = 5",
            "c = a + b",
        ]) + "\n"

    def test_function_sum(self, kwargs):
        """Test the function sum([a, b])."""
        prog = Program()
        # c = sum([a, b])
        a_node = DepNode(type="int", identifier_name="a")
        b_node = DepNode(type="int", identifier_name="b")
        c_node = DepNode(type="int", identifier_name="c")
        list_node = ListOp(op_symbol_or_name="", input_nodes=[a_node, b_node], output_nodes=None)
        sum_node = PrefixOp(
            function_name="sum",
            input_nodes=[list_node],
            output_nodes=[c_node])
        # check that the input of sum are compatible
        assert len(sum_node.input_nodes) == 1
        assert isinstance(sum_node.input_nodes[0], ListOp)
        inner_list_node = sum_node.input_nodes[0]
        assert all([n.type == "int" for n in inner_list_node.input_nodes])
        prog.safe_append_op(sum_node)

        prompt_text = prog.get_full_output()

        assert prompt_text == "\n".join([
            "c = sum([a, b])",
        ]) + "\n"

    def test_get_att_weights_all_relevant(self, kwargs):
        """Derive the attention weights of a specific node."""
        prog = Program()
        # name = "marco"
        name_node = DepNode(type="str", identifier_name="name")
        const_node = DepNode(type="str", identifier_name="marco", literal=True)
        declaration = AssignmentOp(input_nodes=[const_node], output_nodes=[name_node])
        prog.safe_append_op(declaration)
        # surname = "polo"
        surname_node = DepNode(type="str", identifier_name="surname")
        const_node = DepNode(type="str", identifier_name="polo", literal=True)
        declaration = AssignmentOp(input_nodes=[const_node], output_nodes=[surname_node])
        prog.safe_append_op(declaration)
        # full_name = name + surname
        full_name_node = DepNode(type="str", identifier_name="full_name")
        sum_node = InfixOp(
            op_symbol="+",
            input_nodes=[name_node, surname_node],
            output_nodes=[full_name_node])
        prog.safe_append_op(sum_node)

        # print(full_name_node) # prints the value "
        query = QueryOp(
            query_statement=kwargs["custom_prompt"],
            relevant_variable=full_name_node)
        prog.safe_append_op(query)

        prompt_text = prog.get_full_output()
        #print(prompt_text)

        att_weights = prog.get_attention_weights_for(
            identifier=prog.get_id_of_last_relevant_node(),
            distance_fn=convert_distance_to_weight_naive)
        assert np.allclose(att_weights, [
            0.33333333,
            0.33333333,
            0.83333333,
            1.        ])

    def test_get_att_weights_partially_relevant(self, kwargs):
        """Derive the attention weights of a specific node."""
        prog = Program()
        # name = "marco"
        name_node = DepNode(type="str", identifier_name="name")
        const_node = DepNode(type="str", identifier_name="marco", literal=True)
        declaration = AssignmentOp(input_nodes=[const_node], output_nodes=[name_node])
        prog.safe_append_op(declaration)
        # c = min([a, b])
        a_node = DepNode(type="int", identifier_name="a")
        b_node = DepNode(type="int", identifier_name="b")
        c_node = DepNode(type="int", identifier_name="c")
        list_node = ListOp(op_symbol_or_name="", input_nodes=[a_node, b_node], output_nodes=None)
        min_node = PrefixOp(
            function_name="min",
            input_nodes=[list_node],
            output_nodes=[c_node])
        prog.safe_append_op(min_node)
        # print(name_node) # prints the value "
        query = QueryOp(
            query_statement=kwargs["custom_prompt"],
            relevant_variable=name_node)
        prog.safe_append_op(query)

        prompt_text = prog.get_full_output()
        #print(prompt_text)

        att_weights = prog.get_attention_weights_for(
            identifier=prog.get_id_of_last_relevant_node(),
            distance_fn=convert_distance_to_weight_naive)
        assert np.allclose(att_weights, [
            0.6666666666666667,
            0,
            1.0])

    def test_attention_computation_with_derived_dependencies(self, kwargs):
        """The complex program is the following:
        "var_0 = 388",
        "var_1 = 284",
        "var_2 = 552",
        "var_3 = 859",
        "derived_0 = var_0 - var_2",
        "derived_1 = var_2 + var_0",
        "derived_2 = derived_1 * var_1",
        "derived_3 = derived_2 + derived_1",
        "derived_4 = derived_2 / derived_3",
        "if (derived_3 == "
        """
        prog =  torch.load(
            "attwizard/creators/test_assets/simple_derived_program.fixture")
        #prompt_text = prog.get_full_output()
        #print(prompt_text)

        att_weights = prog.get_attention_weights_for(
            identifier=prog.get_id_of_last_relevant_node(),
            distance_fn=convert_distance_to_weight_naive)
        #print(att_weights)
        assert np.allclose(att_weights, [
            0.2222222222222222,
            0.2222222222222222,
            0.2222222222222222,
            0,
            0,
            0.5555555555555556,
            0.5555555555555556,
            0.8888888888888888,
            0,
            1.0])
