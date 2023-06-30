from hashlib import new
from attwizard.creators.prompt_creator import PromptCreator

from typing import List, Dict, Any, Tuple
from attwizard.creators.dep_graph import DepNode, AssignmentOp, Operator, QueryOp, InfixOp, Program
from attwizard.creators.distance_to_weights import convert_distance_to_weight_naive
from random import randint
from random import choice, sample

class AssignmentChainCreator(PromptCreator):

    def __init__(
            self,
            name: str,
            prompt_folder: str, ground_truth_folder: str,
            n_prompts: int = 10,
            attention_level: str = 'line', tokenizer=None,
            kwargs: Dict[str, Any] = None) -> None:
        super().__init__(name, prompt_folder, ground_truth_folder,
                         n_prompts, attention_level, tokenizer, kwargs)

    def create_prompt_and_attention(self) -> Tuple[List[str], List[float]]:
        """Create a new prompt and attention.

        In this case, each part of prompt is a single line.
        """
        prog = Program()
        # generate new assignements
        assignment_ops, available_nodes = self._generate_n_int_assignments(
            n=self.kwargs["n_assignments"],
            min_int=self.kwargs["init_min_int_value"],
            max_int=self.kwargs["init_max_int_value"],)
        prog.append_multiple_ops(assignment_ops)
        # generate arithmetic operations
        arith_ops, derived_nodes = self._generate_n_arithmetic_expression(
            n=self.kwargs["n_arithmetic_ops"],
            available_nodes=available_nodes,
            available_ops=self.kwargs["arithmetic_ops"],
            allow_reuse_of_derived_nodes=self.kwargs["allow_reuse_of_derived_nodes"],
            existing_ops=assignment_ops)
        prog.append_multiple_ops(arith_ops)
        # generate new query
        target_variable = choice(derived_nodes)

        # sanitize the input creating aliases which refer to the
        # most recent assignment of those variables
        #sanitized_new_inputs = self._sanitize_input_by_creating_aliases(
        #    input_nodes=[target_variable],
        #    existing_ops=prog.operations)

        query = QueryOp(
            query_statement=self.kwargs["custom_prompt"],
            relevant_variable=target_variable)
        prog.safe_append_op(query)
        # generate the whole prompt
        prompt_text = prog.get_full_output()
        prompt_parts = prompt_text.split("\n")
        # remove empty lines
        prompt_parts = [x for x in prompt_parts if x != ""]
        # pick the distance function
        if self.kwargs["distance_fn"] == "naive":
            distance_fn = convert_distance_to_weight_naive

        # generate new ground truth
        att_weights = prog.get_attention_weights_for(
            identifier=prog.get_id_of_last_relevant_node(),
            distance_fn=distance_fn)
        # return the prompt and attention
        #import torch
        #torch.save(prog, f"data/ground_truth_attention/exp_v05/{str(prompt_parts[0])}.pt")
        return prompt_parts, att_weights


    def _generate_n_int_assignments(self,
            n: int,
            min_int: int,
            max_int: int,
            existing_ops: List[Operator] = []) -> List[DepNode]:
        """Generate assignment nodes for n integer variables.

        Note that it returns the nodes referring to the just-initialized
        variables."""
        new_assignments = []
        new_assigned_nodes = []
        for i in range(n):
            id_name = DepNode(type="int", identifier_name=f"var_{i}")
            const_val = DepNode(
                type="int", identifier_name=f"{randint(min_int, max_int)}", literal=True)
            new_assignments.append(AssignmentOp(
                input_nodes=[const_val], output_nodes=[id_name]))
            new_assigned_nodes.append(id_name)
        return new_assignments, new_assigned_nodes

    def _generate_n_arithmetic_expression(
            self,
            n: int,
            available_nodes: List[DepNode],
            available_ops: List[Dict[str, Any]],
            allow_reuse_of_derived_nodes: bool = True,
            existing_ops: List[Operator] = []) -> List[DepNode]:
        """Generate n arithmetic expressions."""
        new_arithmetic_expr = []
        new_derived_outputs = []
        c_available_nodes = available_nodes.copy()
        existing_ops = existing_ops.copy()

        for i in range(n):
            # pick one operation
            arith_op_info = choice(available_ops)
            # pick a number of arguments
            n_args = randint(
                arith_op_info["min_n_args"], arith_op_info["min_n_args"])
            # pick n inputs
            input_nodes = sample(c_available_nodes, k=n_args)
            # sanitize the input creating aliases which refer to the
            # most recent assignment of those variables
            #new_input_nodes = self._sanitize_input_by_creating_aliases(
            #    input_nodes=input_nodes,
            #    existing_ops=existing_ops)
            new_node = DepNode(type="int", identifier_name=f"derived_{i}")
            arithm_op = InfixOp(
                op_symbol=arith_op_info["symbol"],
                input_nodes=input_nodes,
                output_nodes=[new_node])
            if allow_reuse_of_derived_nodes:
                c_available_nodes.append(new_node)
            new_derived_outputs.append(new_node)
            new_arithmetic_expr.append(arithm_op)
            existing_ops.append(arithm_op)
        return new_arithmetic_expr, new_derived_outputs


    # def _sanitize_input_by_creating_aliases(
    #         self,
    #         input_nodes: List[DepNode],
    #         existing_ops: List[Operator]) -> List[DepNode]:
    #     """Replace the input nodes with aliases instead of the orginal nodes.

    #     Note that the newly created aliases will point at the orginal nodes and
    #     they will have the same fields except for the id which is always
    #     unique.
    #     """
    #     # get all the outputs of the previous statements
    #     existing_output_nodes = []
    #     for op in existing_ops:
    #         existing_output_nodes.extend(op.output_nodes)
    #     #print("existing_output_nodes: ", [str(n) for n in existing_output_nodes])
    #     # for each input node
    #     # check if in the previous statements/operator,
    #     # the node was used as output (which corresponds to a definition)
    #     new_input_nodes = []
    #     #print("Input nodes: ", [str(n) for n in input_nodes])
    #     for node in input_nodes:
    #         # check if the node is in the list of outputs
    #         # iterate backwards on all the statements until we find
    #         # an alias of the current node in the output of any of
    #         # those statments and return the last def-node.
    #         last_def_node = None
    #         for op in existing_ops[::-1]:
    #             for output_node in op.output_nodes:
    #                 if output_node.is_alias_of(node):
    #                     last_def_node = output_node
    #                     break
    #             if last_def_node is not None:
    #                 break
    #         if last_def_node:
    #             # if yes, create a new node that is a copy of the original node
    #             # but with a different id, then connect the two nodes
    #             # via the alias field in the new node.
    #             # (aka the new node points to the original node)
    #             # get the last definition of the node
    #             new_node = DepNode(
    #                 type=last_def_node.type,
    #                 identifier_name=last_def_node.identifier_name,
    #                 literal=last_def_node.literal)
    #             new_node.add_alias(last_def_node)
    #             new_input_nodes.append(new_node)
    #         else:
    #             # if no, use the existing node.
    #             new_input_nodes.append(node)
    #     return new_input_nodes

