"""This file simplifies the EDA of the realtionships between variables."""

from turtle import title
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Union, Any
from attwizard.creators.test_assignment_chain import kwargs

from attwizard.decoder import get_attention_representation
from attwizard.aligner import map_to_char_level
from attwizard.aligner import tokenize_char_level

from codeattention.source_code import SourceCode


def visualize_corr_vs(
        df: pd.DataFrame,
        col_to_inspect: str,
        var_of_interest: str = "correlation"):
    """Check the relationship between correlation and the given variable."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    ax_hist = axes[0]
    ax_violin = axes[1]
    # HISTGRAM
    sns.histplot(
        data=df,
        x=var_of_interest,
        multiple="stack",
        hue=col_to_inspect,
        ax=ax_hist,
    )
    ax_hist.axvline(0, color="red", linestyle="--")
    ax_hist.set_xlim(-1, 1)
    # VIOLINPLOT
    sns.violinplot(
        data=df,
        x=var_of_interest,
        y=col_to_inspect
    )
    ax_violin.set_xlim(-1, 1)
    plt.tight_layout()


def visualize_code_heatmap_codegen(
        filename: str,
        all_machine_data_vectors: Dict[str, Any],
        all_machine_data_metadata: Dict[str, Any],
        percentile: float = None,
        color: str = "blue",
        distribution_fn: str = "equal_share"):
    """Create the code heatmap starting from the CodeGen model.

    distribution_fn: it can be "equal_share" or "replication"
    """
    c_metadata = all_machine_data_metadata[filename]
    c_data = all_machine_data_vectors[filename]

    # convert the attntion to char level (easier to compare and display)
    att_weights_char_level = map_to_char_level(
        tokens=c_metadata["tokens_prompt"],
        att_weights=c_data,
        raw_text=c_metadata["text_prompt"],
        distribution_fn=distribution_fn,
        special_char_mapping={
            "Ġ": ' ',
            "Ċ": '\n',
        }
    )
    sorted_weights = np.sort(att_weights_char_level)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        np.linspace(0, 1, len(sorted_weights), endpoint=False),
        sorted_weights,
    )
    ax.set_xlabel("(sorted) attention weights")
    ax.set_ylabel("cumulative distribution")

    # cap to percentile value if present
    if percentile is not None:
        max_val = np.percentile(att_weights_char_level, percentile)
        att_weights_char_level = np.clip(
            att_weights_char_level,
            a_min=None,
            a_max=max_val
        )
        # horizontal line
        ax.axhline(max_val, color="red", linestyle="--")
    plt.show()

    char_tokens = tokenize_char_level(c_metadata["text_prompt"])
    # display
    python_sc = SourceCode(char_tokens)
    fig, ax = python_sc.show_with_weights(
        weights=att_weights_char_level,
        show_line_numbers=True,
        char_height=12,
        named_color=color,
    )
    print('This considers only the prompt itself, not the generated code.')
    return fig, ax


def plot_histogram_in_parallel(
        histogram_infos: List[Dict[str, Any]],
        col_name: str,
        common_prefix: str = "",
        scale: str = None,
        xlim: Tuple[float, float] = None,
        height_per_histogram: float = 1.23,
        n_cols: int = 1,
        filling_direction: str = "vertical",
        width: float = 5,
        vertical_zero_line: bool = False,
        show_mean: bool = True,
        show_median: bool = True,
        sharey : bool = False,
        fmt: str = ".2f",
        bins: np.ndarray = None,
        out_path: str = None,
        data_attribute: str = "data"):
    """Plots histograms one below the other with the same scale.

    Parameters:
    -----------
    histogram_infos: list of dict with the key determined by `data_attribute`,
        this filed must contain a pandas dataframe with the data to plot.
    col_name: the column name to plot
    common_prefix: a common prefix to add to the xlabel of each histogram
    scale: the scale to use, it can be "linear" or "log"
    xlim: the limits of the x axis
    height_per_histogram: the height of each histogram ax
    n_cols: the number of columns to use in the figure
    filling_direction: the direction in which the axes in the figure are
        iterated to fill the figure. It can be "vertical" or "horizontal",
        the default is "vertical", meaning that each column is filled
        entirely before moving to the next one.
    width: the width of the entrie figure
    vertical_zero_line: if True, a vertical line is added at x=0
    show_mean: if True, the mean is shown in the histogram
    show_median: if True, the median is shown in the histogram
    sharey: if True, the y axis is shared among all the histograms
    fmt: the format to use for the mean and median
    bins: the bins to use for the histogram
    out_path: if not None, the figure is saved to the given path
    data_attribute: the attribute of the dict in `histogram_infos` that
        contains the data to plot.
    """
    n_rows = int(np.ceil(len(histogram_infos) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width, height_per_histogram * n_rows),
        sharey=sharey,
        squeeze=False,
    )

    for i, info_dict in enumerate(histogram_infos):
        if filling_direction == "vertical":
            row = i % n_rows
            col = i // n_rows
        elif filling_direction == "horizontal":
            row = i // n_cols
            col = i % n_cols
        ax = axes[row][col]
        df_to_show = info_dict[data_attribute]
        kwargs = {}
        if bins is not None:
            kwargs["bins"] = bins
            df_to_show = df_to_show[
                (df_to_show[col_name] >= bins[0]) &
                (df_to_show[col_name] <= bins[-1])
            ]
        sns.histplot(
            data=df_to_show,
            x=col_name,
            color=info_dict["color"],
            ax=ax,
            **kwargs
        )
        ax.set_xlabel(f"{common_prefix} {info_dict['long']}")
        legend_patches = []
        # plot a vertical line for mean and median, in red and green
        if show_median:
            median = info_dict[data_attribute][col_name].median()
            ax.axvline(median, color='green')
            legend_patches.append(f'Median: {median:{fmt}}')
        if show_mean:
            mean = info_dict[data_attribute][col_name].mean()
            ax.axvline(mean, color='red', linestyle='--')
            legend_patches.append(f"Mean: {mean:{fmt}}")
        ax.legend(legend_patches)
        if vertical_zero_line:
            ax.axvline(0, color='black', linestyle='--')
        if scale:
            ax.set_xscale(scale)
        if xlim:
            ax.set_xlim(xlim)
        plt.tight_layout()
    if out_path:
        fig.savefig(out_path)
    return fig, axes


def plot_100_perc_stacked_bar(
        df: pd.DataFrame,
        col_name_group: str,
        col_name_category_in_group: str,
        direction: str = "vertical",
        group_order: List[str] = None,
        n_cols_legend: int = 3,
        legend_title: str = "",
        fmt: str = "{:.0%}",
        convert_x_to_perc: bool = True,
        ax: plt.Axes = None,
        **kwargs: Any):
    """Plot a 100% stacked bar chart from a dataframe.

    Parameters:
    -----------
    df: the dataframe to plot. It must have the following columns:
        - `col_name_group`: the name of the group
        - `col_name_category_in_group`: the name of the category in the group
    col_name_group: the name of the column that contains the group name
    col_name_category_in_group: the name of the column that contains the
        categories which should make up the percentages in the stacked bar
    direction: the direction in which the bars are stacked. It can be
        "vertical" or "horizontal", the default is "vertical".
    """
    df_grouped = df.groupby(col_name_group)[
        col_name_category_in_group].value_counts(normalize=True).unstack(
            col_name_category_in_group)
    assert direction in ["vertical", "horizontal"], (
        f"direction must be 'vertical' or 'horizontal', not {direction}")
    if ax:
        kwargs["ax"] = ax
    if group_order is not None:
        mapping = {name: i for i, name in enumerate(group_order)}
        key = df_grouped.index.map(mapping)
        df_grouped = df_grouped.iloc[key.argsort()]
    if direction == "vertical":
        # set colors for each category red, green, blue, yellow
        ax = df_grouped.plot.bar(stacked=True, **kwargs)
    elif direction == "horizontal":
        ax = df_grouped.plot.barh(stacked=True, **kwargs)
    ax.legend(
        title=legend_title,
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
        borderaxespad=0,
        frameon=False,
        ncol=n_cols_legend,
    )
    for ix, row in df_grouped.reset_index(drop=True).iterrows():
        cumulative = 0
        for element in row:
            if element > 0.1:
                if direction == "vertical":
                    coord = (ix, cumulative + element / 2)
                elif direction == "horizontal":
                    coord = (cumulative + element / 2, ix)
                ax.text(
                    *coord,
                    fmt.format(element),
                    va="center",
                    ha="center",
                )
            # check if element is pandas nan
            if str(element) == "nan":
                continue
            cumulative += element
    #replace ticks with the percentage
    if convert_x_to_perc:
        if direction == "horizontal":
            ax.set_xticklabels(
                [f"{int(x * 100)} %" for x in ax.get_xticks()],
                rotation=0,
            )
            ax.grid(axis="x")
        elif direction == "vertical":
            ax.set_yticklabels(
                [f"{int(x * 100)} %" for x in ax.get_yticks()],
                rotation=0,
            )
            # switch on the grid on this axis
            ax.grid(axis="y")
    # get figure
    fig = ax.get_figure()
    return fig, ax
