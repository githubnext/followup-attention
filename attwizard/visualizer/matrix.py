"""This file plots the attention matrices."""

import torch
import numpy as np
import networkx as nx
from typing import ForwardRef, List, Tuple, Any, Dict, Union
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import time
from tqdm import tqdm


def heatmap_visualize(
        attention_matrix: Union[np.ndarray, ForwardRef('torch.Tensor')],
        token_names: List[str],
        n_input_tokens: int = 0,
        ax=None,
        figsize=(10, 10),
        cmap: str = 'Reds'):
    """Visualize the attention matrix as heatmap."""
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    sns.heatmap(
        attention_matrix,
        xticklabels=token_names,
        yticklabels=token_names,
        cmap=cmap,
        ax=ax,
    )
    plt.tight_layout()
    if n_input_tokens > 0:
        # draw a rectangle in the top right corner
        ax.add_patch(
            Rectangle(
                (0, 0),
                n_input_tokens, n_input_tokens,
                fill=False,
                edgecolor="black",
                linewidth=1,
            )
        )
    return fig, ax


def visualize_slices_of_3D_tensor(
        tensor_3d: Union[np.ndarray, ForwardRef('torch.Tensor')],
        axis: int = 0,
        cutoff_percentile: float = 99.9,
        titles: List[str] = None,
        x_tick_labels: List[str] = None,
        y_tick_labels: List[str] = None,
        x_tick_skip_ratio: float = 0.5,
        y_tick_skip_ratio: float = 0.5,
        highlight_area: Tuple[int, int] = None,
        graphic_tile_size: Tuple[int, int] = (4, 4),
        max_cols: int = 5,
        cmap: str = 'Reds',):
    """Visualize slices of a 3D tensor.

    This method creates a grid of tiles, where each tile contains a heatmap
    of the slice of the tensor. The slices are taken along the given axis.

    Parameters
    ----------
    tensor_3d : the 3D tensor to visualize
    cutoff_percentile : the percentile to cut off the values of the tensor
        if they are above this threshold (they will get the percentile value)
    titles: the list of titles for each tile
    x_ticklabels : the ticklabels for the x-axis
    y_ticklabels : the ticklabels for the y-axis
    x_tick_skip_ratio : the ratio of ticks to skip on the x-axis
    y_tick_skip_ratio : the ratio of ticks to skip on the y-axis
    highlight_area : the area to highlight in the heatmap with a square box
        starting on the top left corner
    graphic_tile_size : the size of each tile in inches, first value is
        width, second value is height
    max_cols : the maximum number of columns in the grid
    cmap : the color map to use for the heatmap
    """
    n_layers = tensor_3d.shape[axis]
    n_rows = np.ceil(n_layers / max_cols).astype(int)
    fig, axes = plt.subplots(
        n_rows, max_cols,
        figsize=(
            graphic_tile_size[0] * max_cols, graphic_tile_size[1] * n_rows))
    for i_layer, i_title in tqdm(zip(range(n_layers), titles)):
        i_row = int(i_layer / max_cols)
        i_col = i_layer % max_cols
        i_ax = axes[i_row][i_col]
        if axis == 0:
            i_layer_matrix = tensor_3d[i_layer]
        elif axis == 1:
            i_layer_matrix = tensor_3d[:, i_layer, :].squeeze()
        elif axis == 2:
            i_layer_matrix = tensor_3d[:, :, i_layer].squeeze()
        percentile = np.percentile(i_layer_matrix, cutoff_percentile)
        i_layer_matrix[i_layer_matrix > percentile] = percentile
        # skip some ticks
        if x_tick_skip_ratio < 1:
            x_tick_labels = [
                tick if i % int(1 / (1 - x_tick_skip_ratio)) == 0 else ''
                for i, tick in enumerate(x_tick_labels)]
        if y_tick_skip_ratio < 1:
            y_tick_labels = [
                tick if i % int(1 / (1 - y_tick_skip_ratio)) == 0 else ''
                for i, tick in enumerate(y_tick_labels)]
        sns.heatmap(
            i_layer_matrix,
            xticklabels=x_tick_labels,
            yticklabels=y_tick_labels,
            cmap=cmap,
            ax=i_ax,
        )
        i_ax.set_title(i_title)
        if highlight_area:
            i_ax.add_patch(
                Rectangle(
                    (0, 0),
                    highlight_area[0], highlight_area[1],
                    fill=False,
                    edgecolor="black",
                    linewidth=1,
                )
            )
    plt.tight_layout()
    return fig, axes


def visualize_layer_wise_attention(
        adj_mat: Union[np.ndarray, ForwardRef('torch.Tensor')],
        mapping_node_label_to_token_pos: Dict[str, int],
        n_layers: int,
        n_tokens: int,
        n_bin_weights: int = 0,
        ax=None,
        no_output: bool = False,):
    """Visualize the attention matrix as network, layer by layer.

    Parameters:
        adj_mat: np.ndarray, the adjacency matrix of the graph.
        mapping_node_label_to_token_pos: Dict[str, int], the mapping from
            labels of a node to position index of the token in the sequence,
            namely on which row each node should appear.
        n_layers: int, the number of layers in the graph (as in the model).
        n_tokens: int, the number of tokens in the attended sequence.
        n_bin_weights: int, the number of bins to use to split the edges.
            If 0 no binning is done and the precise weights are used.
            E.g. if bin 10, all the weights will be divided in 10 bins and
            each of them will receive only one value of those 10, namely the
            closest one.


    Returns:
        G: nx.Graph, the graph showing the attention over multiple layers.
        ax: matplotlib.axes.Axes, the axes of the plot.

    Credits:
    Inspired by code from:
    “Quantifying Attention Flow In Transformers”, ACL 2020
    """
    A = adj_mat

    if n_tokens > 20:
        print(
            "Warning: with high number of tokens it could take long. " +
            "Be patient.")

    # Bin-arize the weights
    if n_bin_weights > 0:
        max_weight = A.max()
        min_weight = A.min()
        lin_space = np.linspace(min_weight, max_weight, num=n_bin_weights)
        A_ditialized = np.digitize(A, bins=lin_space)
        A_ditialized = A_ditialized - 1
        A = np.array(list(map(
            lambda e: lin_space[e], A_ditialized)))
    print(time.time())
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i, j): A[i, j]}, 'capacity')

    pos = {}
    label_pos = {}
    print(time.time())
    for i in np.arange(n_layers+1):
        for k_f in np.arange(n_tokens):
            pos[i*n_tokens+k_f] = ((i+0.5)*2, n_tokens - k_f)
            label_pos[i*n_tokens+k_f] = (i*2, n_tokens - k_f)

    print(time.time())
    index_to_labels = {}
    for key in mapping_node_label_to_token_pos.keys():
        index_to_labels[mapping_node_label_to_token_pos[key]] = \
            key.split("_")[-1]
        if mapping_node_label_to_token_pos[key] >= n_tokens:
            index_to_labels[mapping_node_label_to_token_pos[key]] = ''

    if not no_output:
        if not ax:
            fig, ax = plt.subplots(figsize=(20, 12))

        nx.draw_networkx_nodes(
            G, pos, node_color='black', node_size=50)
        nx.draw_networkx_labels(
            G, pos=label_pos, labels=index_to_labels, font_size=10)

        all_weights = []
        # 4 a. Iterate through the graph nodes to gather all the weights
        for (node1, node2, data) in G.edges(data=True):
            # the edge thickness will depend on the weights
            all_weights.append(data['weight'])

        # 4 b. Get unique weights
        unique_weights = list(set(all_weights))

        # 4 c. Plot the edges - one by one!
        for weight in unique_weights:
            # 4 d. Form a filtered list with just the weight you want to draw
            weighted_edges = [
                (node1, node2) for (node1, node2, edge_attr) in G.edges(data=True)
                if edge_attr['weight'] == weight]
            # 4 e. I think multiplying by [num_nodes/sum(all_weights)]
            # makes the graphs edges look cleaner

            w = weight
            # alternative
            # w = (weight - min(all_weights))/(max(all_weights) - min(all_weights))
            width = w
            nx.draw_networkx_edges(
                G, pos, edgelist=weighted_edges,
                width=width, edge_color='red',
                ax=ax)

    if no_output:
        ax = None
    return G, ax


def visualize_followup_graph_side_by_side(
        adj_mat: Union[np.ndarray, ForwardRef('torch.Tensor')],
        from_seq: List[str],
        to_seq: List[str],
        show_only_top_k: int = 3,
        max_length_per_label: int = 50,
        multiline_labels: bool = True,
        vertical_elements_spacing: Tuple[int, int] = (.3, .3),
        fig_size_width: int = 15,
        divide_by_max_val_matrix: bool = False,
        ax=None):
    """Visualize the connections between two sets of entities.

    These entities are tipically:
    1. token list vs (same) token list
    2. list of lines vs (same) list of lines
    3. token list vs (corespoding) list of lines
    4. list of lines vs (corresponding) token list

    Parameters:
    - adj_mat: np.ndarray, the adjacency matrix of the graph.
    - from_seq: List[str], the list of entities from which the connections
        are coming from. Note that each corresponds to a row of the adjacency
        matrix.
    - to_seq: List[str], the list of entities to which the connections are
        going to. Note that each corresponds to a column of the adjacency
        matrix.
    - show_only_top_k: int, the number of connections to show.
    - max_length_per_label: int, the maximum length of the labels.
    - multiline_labels: bool, whether to show the labels in multiple lines.
    - vertical_elements_spacing: Tuple[int, int], the spacing between the
        elements, the first number is for the `from` nodes and the second for the
        `to` nodes.
    - fig_size_width: int, the width of the figure.
    - ax: matplotlib.axes.Axes, the axes of the plot.
    """
    figsize = (
        fig_size_width, max(
            len(from_seq) * vertical_elements_spacing[0],
            len(to_seq) * vertical_elements_spacing[1]
        )
    )

    G = nx.DiGraph()
    pos = {}

    def convert_to_multiline(in_string: str, max_char_per_line: int) -> str:
        """Convert a single line to a multiline string."""
        out_string = ""
        for i in range(0, len(in_string), max_char_per_line):
            out_string += in_string[i:i+max_char_per_line] + "\n"
        return out_string

    if multiline_labels:
        from_seq = [
            l if len(l) < max_length_per_label
            else convert_to_multiline(l, max_length_per_label)
            for l in from_seq
        ]
        to_seq = [
            l  if len(l) < max_length_per_label
            else convert_to_multiline(l, max_length_per_label)
            for l in to_seq
        ]

    max_val = 1
    if divide_by_max_val_matrix:
        max_val = adj_mat.max()

    # add destination node (aka target lines)
    for i in range(len(to_seq)):
        node_id = f"line_{i}_dest"
        G.add_node(node_id, label=to_seq[i], color="red")
        pos[node_id] = (1, - i * vertical_elements_spacing[1])

    # add starting nodes + top k followup edges
    for i in range(len(from_seq)):
        top_k = torch.argsort(adj_mat[i, :])[-show_only_top_k:]
        top_k = top_k.tolist()[::-1]
        node_id = f"line_{i}_src"
        G.add_node(node_id, label=from_seq[i], color="blue")
        pos[node_id] = (0, - i * vertical_elements_spacing[0])
        for connection_rank, j in enumerate(top_k):
            # print(f"Rank {connection_rank}: " +
            # "{followup_line_matrix[i, j]/max_val}")
            target_node_id = f"line_{j}_dest"
            if connection_rank == 0:
                edge_color = "dodgerblue"
            elif connection_rank == 1:
                edge_color = "darkorange"
            else:
                edge_color = "forestgreen"
            G.add_edge(
                node_id, target_node_id,
                weight=adj_mat[i, j]/max_val, color=edge_color)

    # draw graph with custom positions
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    node_labels = nx.get_node_attributes(G, 'label')
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u,v in edges]
    edge_weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw(G, pos, with_labels=True, labels=node_labels,
        edge_color=edge_colors, width=edge_weights)
    # add legend
    legend_elements = [
        Patch(
            facecolor='dodgerblue', edgecolor='black', label='1st Connection'),
        Patch(
            facecolor='darkorange', edgecolor='black', label='2nd Connection'),
        Patch(
            facecolor='forestgreen', edgecolor='black', label='3nd Connection'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.margins(x=0.4)
    return fig, ax

