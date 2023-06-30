
from typing import Callable, List, Dict, Tuple, Union
import random
import networkx as nx
from networkx import descendants_at_distance, all_neighbors
from matplotlib import pyplot as plt
from attwizard.creators.distance_to_weights import convert_distance_to_weight_naive


class DepNode(object):

    def __init__(
            self,
            type: str,
            identifier_name: str,
            literal: bool = False) -> None:
        self.type = type
        self.identifier_name = identifier_name
        self.literal = literal
        self.alias = None
        self.id = str(random.randint(0, 10000000000))

    def __str__(self):
        if self.type == 'int' or self.type == "float":
            return self.identifier_name
        if self.literal:
            return f"'{self.identifier_name}'"
        return self.identifier_name

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.id == other.id
        return False

    def add_alias(self, alias_node):
        if isinstance(alias_node, type(self)):
            if self.is_alias_of(alias_node):
                self.alias = alias_node
            else:
                raise ValueError(f"The node {str(alias_node)} is not a valid alias of {str(self)}")
        else:
            raise TypeError(f"The alias {str(alias_node)} must be a DepNode. Instead, type {type(alias_node)} was given.")

    def is_alias_of(self, other):
        if isinstance(other, type(self)):
            return self.type == other.type and \
                self.identifier_name == other.identifier_name and \
                self.literal == other.literal
        return False

    def __repr__(self):
        return f"({self.type}) {self.identifier_name} (lit:{self.literal})"

class Operator(object):
    """This represents either an arithmetic operation or a function call."""

    def __init__(
            self,
            op_symbol_or_name: str,
            input_nodes: List[DepNode],
            output_nodes: List[DepNode],
            side_effect_nodes: List[DepNode] = None) -> None:
        self.op_symbol_or_name = op_symbol_or_name
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.side_effect_nodes = side_effect_nodes
        self.id = random.randint(0, 10000000000)

    def __str__(self) -> str:
        representation = ""
        representation += f"{self.op_symbol_or_name}:\n"
        representation += f"- INPUT: {', '.join([str(n) for n in self.input_nodes])}\n"
        representation += f"- OUTPUT: {', '.join([str(n) for n in self.output_nodes])}\n"
        return representation

    def str_with_output(self) -> str:
        return f"{', '.join([str(n) for n in self.output_nodes])} = " + self.__str__()


class ListOp(Operator):

    def __init__(
            self,
            op_symbol_or_name: str,
            input_nodes: List[DepNode], output_nodes: List[DepNode],
            side_effect_nodes: List[DepNode] = None) -> None:
        super().__init__(
            op_symbol_or_name, input_nodes, output_nodes, side_effect_nodes)

    def __str__(self) -> str:
        separator = ", "
        return f"[{separator.join([str(n) for n in self.input_nodes])}]"


class AssignmentOp(Operator):

    def __init__(
            self,
            input_nodes: List[DepNode],
            output_nodes: List[DepNode],
            side_effect_nodes: List[DepNode] = None) -> None:
        super().__init__(
            "=", input_nodes, output_nodes, side_effect_nodes)

    def str_with_output(self) -> str:
        return f"{', '.join([str(n) for n in self.output_nodes])} = {', '.join([str(n) for n in self.input_nodes])}"


class PrefixOp(Operator):
    """This represents a prefix operator such as a function call."""

    def __init__(
            self,
            function_name: str,
            input_nodes: List[DepNode],
            output_nodes: List[DepNode],
            side_effect_nodes: List[DepNode] = None) -> None:
        super().__init__(
            function_name, input_nodes, output_nodes, side_effect_nodes)

    def __str__(self) -> str:
        return f"{self.op_symbol_or_name}({', '.join([str(n) for n in self.input_nodes])})"


class InfixOp(Operator):
    """This represents an infix operator such as arithmetic or bool ops."""

    def __init__(
            self,
            op_symbol: str,
            input_nodes: List[DepNode],
            output_nodes: List[DepNode],
            side_effect_nodes: List[DepNode] = None) -> None:
        super().__init__(
            op_symbol, input_nodes, output_nodes, side_effect_nodes)

    def __str__(self) -> str:
        op_separator = " " + self.op_symbol_or_name + " "
        return f"{op_separator.join([str(n) for n in self.input_nodes])}"


class QueryOp(Operator):
    """This represents the last staement to stimulate the language model."""

    def __init__(
            self,
            query_statement: str,
            relevant_variable: DepNode,
            ) -> None:
        super().__init__(
            query_statement, [relevant_variable], [], side_effect_nodes = None)

    def str_with_output(self) -> str:
        return self.op_symbol_or_name.replace(
            "<RELEVANT_VARIABLE>", self.input_nodes[0].identifier_name)


class Program(object):

    """This object keeps track of the order in which the ops are performed."""

    def __init__(self) -> None:
        self.operations = []  # type: List[Operator]
        self.G = None
        self.pos = None

    def safe_append_op(self, operation: Operator) -> None:
        """Append a new operation by sanitizing the inputs.

        It replace the input nodes which were have already been defined
        with new aliases which point to them.
        """
        old_input_nodes = operation.input_nodes
        new_input_nodes = self._sanitize_input_by_creating_aliases(
            input_nodes=old_input_nodes,
            existing_ops=self.operations)
        operation.input_nodes = new_input_nodes
        self.operations.append(operation)

    def _sanitize_input_by_creating_aliases(
            self,
            input_nodes: List[DepNode],
            existing_ops: List[Operator]) -> List[DepNode]:
        """Replace the input nodes with aliases instead of the orginal nodes.

        Note that the newly created aliases will point at the orginal nodes and
        they will have the same fields except for the id which is always
        unique.
        """
        # get all the outputs of the previous statements
        existing_output_nodes = []
        for op in existing_ops:
            existing_output_nodes.extend(op.output_nodes)
        #print("existing_output_nodes: ", [str(n) for n in existing_output_nodes])
        # for each input node
        # check if in the previous statements/operator,
        # the node was used as output (which corresponds to a definition)
        new_input_nodes = []
        #print("Input nodes: ", [str(n) for n in input_nodes])
        for node in input_nodes:
            # check if the node is in the list of outputs
            # iterate backwards on all the statements until we find
            # an alias of the current node in the output of any of
            # those statments and return the last def-node.
            last_def_node = None
            for op in existing_ops[::-1]:
                for output_node in op.output_nodes:
                    if output_node.is_alias_of(node):
                        last_def_node = output_node
                        break
                if last_def_node is not None:
                    break
            if last_def_node:
                # if yes, create a new node that is a copy of the original node
                # but with a different id, then connect the two nodes
                # via the alias field in the new node.
                # (aka the new node points to the original node)
                # get the last definition of the node
                new_node = DepNode(
                    type=last_def_node.type,
                    identifier_name=last_def_node.identifier_name,
                    literal=last_def_node.literal)
                new_node.add_alias(last_def_node)
                new_input_nodes.append(new_node)
            else:
                # if no, use the existing node.
                new_input_nodes.append(node)
        return new_input_nodes

    def append_multiple_ops(self, ops: List[Operator]) -> None:
        for op in ops:
            self.safe_append_op(op)

    def get_full_output(self) -> str:
        """Reconstruct the textual version of each operator and concatenate."""
        output_text = ""
        for op in self.operations:
            output_text += op.str_with_output() + "\n"
        return output_text

    def get_id_of_last_relevant_node(self) -> str:
        """Get the id of the last prompt."""
        last_node = self.operations[-1].input_nodes[0]
        return last_node.id

    def get_attention_weights_for(
            self,
            identifier: str,
            distance_fn: Callable[[int, int, int], float] = None) -> List[float]:
        """Get the attention weights for the given identifier.

        The identifier can either be the identifier name or the numeric id of
        the node.
        We assume that all the nodes on which the identifier name depends are
        relevant and get 1 weight for each of them, whreas the other (not
        involved statements) get 0.
        """
        attention_weights = []
        if isinstance(identifier, str):
            target_node = identifier
        else:
            raise TypeError(
                f"Expected identifier to be str, got `{str(identifier)}` of type{type(identifier)}")
        related_nodes = self.get_dependency_tree(target_node)
        # print("related_nodes:", related_nodes)
        n_lines = len(self.operations)
        max_depth = max([n["depth"] for n in related_nodes])
        for i in range(1, n_lines + 1):
            # print("on line:", i)
            related_nodes_on_line = [
                n for n in related_nodes if n["line_number"] == i]
            # print("there are the following nodes:", related_nodes_on_line)
            if len(related_nodes_on_line) > 0:
                min_depth_on_line = min(
                    [n["depth"] for n in related_nodes_on_line])
            else:
                min_depth_on_line = None
            att_weight = distance_fn(
                distance=min_depth_on_line,
                max_distance=max_depth,
                n_lines=n_lines,
            )
            attention_weights.append(att_weight)
        return attention_weights

    def get_node_ids_for(self, identifier_name: str) -> List[int]:
        """Return the node ids of a given identifier.

        Note that it can return multiple ids, if there are many nodes with the
        same node name.
        """
        self._ensure_that_graph_exists()
        # print(">>>>> looking for: ", identifier_name)
        # for node_id in self.G.nodes:
        #     print("node_id:", node_id, " - type:", type(node_id))
        #     print("self.G.nodes[node_id]: ", self.G.nodes[node_id])
        # import pdb; pdb.set_trace()
        matched_nodes = [
            node_id
            for node_id in self.G.nodes
            if self.G.nodes[node_id]["label"] == identifier_name]
        return matched_nodes

    def __str__(self) -> str:
        return f"{self.operations}"

    def _ensure_that_graph_exists(self) -> None:
        """Create the graph representation if it doesn't exists yet."""
        if self.G is None:
            self.create_graph()

    def get_dependency_tree(
            self,
            node_id: int,
            max_level_of_depth: int = 10) -> List[Dict[str, int]]:
        """This represents the dependency tree of a specific node.

        Note that this is given as a result of a query on a specific user statment
        (e.g. get_depend_tree_of(print(a)) will return the DepTree of a).
        Each node in the tree has a depth level, to represent how far down the tree
        the node is.

        Return a list of records:
        [
            {
                "node_id": <node_id>,
                "depth": <depth_level>,
                "line_number": <line_number>
            },
            ...
        ]
        """
        self._ensure_that_graph_exists()
        records = []
        for d_distance in range(max_level_of_depth):
            decscendants_at_distance_d = descendants_at_distance(
                self.G, source=node_id, distance=d_distance
            )
            for descendant in decscendants_at_distance_d:
                records.append({
                    "node_id": descendant,
                    "depth": d_distance,
                    "line_number": self.G.nodes[descendant]["line"]
                })
        return records

    def create_graph(self) -> nx.DiGraph:
        """Create a graph from the dependency tree."""
        self.pos_and_operators = list(enumerate(self.operations))
        self.root_node = DepNode(type="ROOT", identifier_name="ROOT")
        G = nx.DiGraph()
        root_id = self.root_node.id
        G.add_node(str(root_id), line=0, color='red', subtype=None, label="ROOT")
        pos = {}
        pos[root_id] = (-5, -len(self.pos_and_operators))
        self._create_graph(
            G,
            parent_node_id=root_id,
            connection_type="ROOT",
            to_parse=self.pos_and_operators,
            pos=pos)

        # remove the root helper for better visualization
        G.remove_node(str(root_id))
        pos.pop(str(root_id), None)
        self.G = G
        self.pos = pos
        return G, pos

    def create_def_use_matrix(self) -> List[List[int]]:
        """Create a matrix of dependencies between operators.

        The matrix is a list of lists, where each row is a variable and each
        column is an operator.
        The value in the matrix is:
        - 1 if the variable is defined in that operator
        - 2 if the variable is used in that operator
        - 0 if the variable is neither used or defined in the operator.
        """
        pass

    def _create_graph(
            self,
            G: nx.DiGraph,
            parent_node_id: str,
            connection_type: str,
            to_parse: Union[List[Tuple[int, Operator]], Tuple[int, Operator]],
            pos: Dict[int, Tuple[int, int]]) -> None:
        """Create the graph by iterating over all operators."""
        if isinstance(to_parse, Tuple) and \
                isinstance(to_parse[0], int) and \
                isinstance(to_parse[1], Operator):
            pos_and_operators = [to_parse]
        else:
            pos_and_operators = to_parse

        for idx_line, op in pos_and_operators:
            # print("op:", str(op))
            idx_line += 1
            # create a node for the operator
            G.add_node(str(op.id), line=idx_line, type="OPERATOR", subtype=str(op.__class__.__name__), color='lightgreen', label=str(op.__class__.__name__))
            pos[str(op.id)] = (0, -2 * idx_line)
            # connect it to the parent node
            G.add_edge(str(parent_node_id), str(op.id), type=connection_type)
            # create a node for the connected nodes
            connection_types = []
            all_connected_nodes = []
            if op.input_nodes:
                connection_types.extend(["INPUT"] * len(op.input_nodes))
                all_connected_nodes.extend(op.input_nodes)
            if op.output_nodes:
                connection_types.extend(["OUTPUT"] * len(op.output_nodes))
                all_connected_nodes.extend(op.output_nodes)
            input_offset = 1
            output_offset = 1
            # print("all_connected_nodes:", [str(n) for n in all_connected_nodes])
            for node, connection_type in (zip(all_connected_nodes, connection_types)):
                if isinstance(node, Operator):
                    # if the input node is an operator,
                    # connect them recursively to their upper node
                    self._create_graph(
                        G,
                        parent_node_id=op.id,
                        connection_type=connection_type,
                        to_parse=(idx_line, node),
                        pos=pos)
                else:
                    #self.add_node_and_connect_to_alias(
                    #    G,
                    #    node,
                    #    idx_line
                    #)
                    G.add_node(str(node.id), line=idx_line, type="VARIABLE", color='yellow', label=str(node))
                    # connect to the alias if any
                    if node.alias:
                        G.add_edge(str(node.id), str(node.alias.id), type="ALIAS")
                    if connection_type == "INPUT":
                        pos[str(node.id)] = (input_offset, -2 * idx_line + 1)
                        input_offset += 1
                        G.add_edge(str(op.id), str(node.id), type="INPUT")
                    elif connection_type == "OUTPUT":
                        pos[str(node.id)] = (-output_offset, -2 * idx_line + 1)
                        output_offset += 1
                        G.add_edge(str(node.id), str(op.id), type="OUTPUT")

    def visualize_graph(self, show_ids=False) -> None:
        """Create a matplotlib graph of the given dependency tree."""
        self._ensure_that_graph_exists()
        color_map = [self.G.nodes[node_id]['color'] for node_id in self.G]
        labels = nx.get_node_attributes(self.G, 'label')
        if show_ids:
            labels = {node_id: f"{labels[node_id]} ({node_id})" for node_id in self.G}
        nx.draw(
            self.G,
            with_labels=True,
            font_weight='bold',
            node_color=color_map,
            pos=self.pos,
            labels=labels)
        plt.gcf().set_size_inches(15, 15)
        plt.margins(x=0.4)


