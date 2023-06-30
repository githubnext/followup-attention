"""Comparison functions to compare the attention data human vs machine.

These functions take the two data from human and machine (e.g. vectors or
matrices) and they return a dictionary containing the result of the comparison.
"""


from scipy.stats import spearmanr
from scipy.stats import kstest
import statistics
from scipy.spatial.distance import jensenshannon
import numpy as np

from transformers import AutoTokenizer
from attwizard.aligner import get_tokens_with_col_and_line
from attwizard.script.utils import get_model_folder_path
from attwizard.script.utils_model import get_tokenization_from_machine_metadata


def get_comp_fn(name):
    """Get the comparison function given its name."""
    if name == "spearman_rank":
        return spearman_rank
    elif name == "mse":
        return mse
    elif name == "mse_line_by_line":
        return mse_line_by_line
    elif name == "jsd_line_by_line":
        return jsd_line_by_line
    elif name == "compare_mean_reciprocal_rank":
        return compare_mean_reciprocal_rank
    elif name == "compare_harmonic_rank":
        return compare_harmonic_rank
    elif name == "spearman_rank_by_line":
        return spearman_rank_by_line
    elif name == "perfect_match":
        return perfect_match
    elif name == "rank":
        return rank
    elif name == "top_10_overlap":
        return top_10_overlap
    elif name == "top_3_overlap":
        return top_3_overlap
    elif name == "tok_k_overlap_far_j":
        return tok_k_overlap_far_j
    else:
        raise ValueError("Unknown comparison function: {}".format(name))


# VECTORS


def spearman_rank(human_data, machine_data):
    """Calculate the spearman rank correlation between the two data."""
    assert human_data.shape == machine_data.shape
    res = spearmanr(human_data, machine_data)
    comp_res = {
        "correlation": res.correlation,
        "pvalue": res.pvalue,
    }
    return comp_res


# MATRICES

def compare_mean_reciprocal_rank(human_data, machine_data):
    """Calculate the reciprocal rank between the two data."""
    assert human_data.shape == machine_data.shape
    n_matrix_lines = human_data.shape[0]
    reciprocal_rank_list = []
    for i in range(n_matrix_lines):
        line_human_data = human_data[i]
        line_machine_data = machine_data[i]
        ranking_human = line_human_data.argsort()[::-1]
        ranking_machine = line_machine_data.argsort()[::-1]
        target_top_answer_for_human = ranking_human[0]
        position_in_machine_rank = \
            list(ranking_machine).index(target_top_answer_for_human)
        reciprocal_rank = 1.0 / (position_in_machine_rank + 1)
        reciprocal_rank_list.append(reciprocal_rank)
    comp_res = {
        "mrr": np.mean(reciprocal_rank_list),
    }
    return comp_res


def compare_harmonic_rank(human_data, machine_data):
    """Compute the harmonic mean of the ranks."""
    assert human_data.shape == machine_data.shape
    n_matrix_lines = human_data.shape[0]
    ranks = []
    for i in range(n_matrix_lines):
        line_human_data = human_data[i]
        line_machine_data = machine_data[i]
        ranking_human = line_human_data.argsort()[::-1]
        ranking_machine = line_machine_data.argsort()[::-1]
        target_top_answer_for_human = ranking_human[0]
        position_in_machine_rank = \
            list(ranking_machine).index(target_top_answer_for_human) + 1
        ranks.append(position_in_machine_rank)
    comp_res = {
        "harmonic_rank": statistics.harmonic_mean(ranks),
    }
    return comp_res


def rank(human_data, machine_data, symmetric=False):
    """Compute the rank of the first suggestin for each line."""
    assert human_data.shape == machine_data.shape
    n_matrix_lines = human_data.shape[0]
    ranks = []
    for i in range(n_matrix_lines):
        line_human_data = human_data[i]
        if np.sum(line_human_data) == 0:
            ranks.append([-1, i])
            continue
        line_machine_data = machine_data[i]
        if symmetric and np.sum(line_machine_data) == 0:
            ranks.append([-1, i])
            continue
        ranking_human = line_human_data.argsort()[::-1]
        ranking_machine = line_machine_data.argsort()[::-1]
        target_top_answer_for_human = ranking_human[0]
        position_in_machine_rank = \
            list(ranking_machine).index(target_top_answer_for_human) + 1
        ranks.append([position_in_machine_rank, i])
    comp_res = {
        "rank_list": ranks,
    }
    return comp_res


def perfect_match(human_data, machine_data, symmetric=False):
    """Check how often their Top-1 is the same.

    Namely when the first rank next position of the model is the same of that
    of the human."""
    assert human_data.shape == machine_data.shape
    n_matrix_lines = human_data.shape[0]
    ranks = []
    for i in range(n_matrix_lines):
        line_human_data = human_data[i]
        if np.sum(line_human_data) == 0:
            ranks.append([-1, i])
            continue
        line_machine_data = machine_data[i]
        if symmetric and np.sum(line_machine_data) == 0:
            ranks.append([-1, i])
            continue
        ranking_human = line_human_data.argsort()[::-1]
        ranking_machine = line_machine_data.argsort()[::-1]
        target_top_answer_for_human = ranking_human[0]
        top_answer_for_machine = ranking_machine[0]
        is_perfect_match = \
            int(target_top_answer_for_human == top_answer_for_machine)
        ranks.append([is_perfect_match, i])
    comp_res = {
        "perfect_match_list": ranks,
    }
    return comp_res


def top_k_overlap(human_data, machine_data, symmetric=False, k=10):
    """Check how many top-k overlaps."""
    assert human_data.shape == machine_data.shape
    n_matrix_lines = human_data.shape[0]
    ranks = []
    for i in range(n_matrix_lines):
        line_human_data = human_data[i]
        if np.sum(line_human_data) == 0:
            ranks.append([-1, i, [], []])
            continue
        line_machine_data = machine_data[i]
        if symmetric and np.sum(line_machine_data) == 0:
            ranks.append([-1, i, [], []])
            continue
        ranking_human = line_human_data.argsort()[::-1]
        ranking_machine = line_machine_data.argsort()[::-1]
        top_k_human = ranking_human[:k]
        top_k_machine = ranking_machine[:k]
        top_k_overlap = len(set(top_k_human) & set(top_k_machine))
        ranks.append([top_k_overlap, i, list(top_k_human), list(top_k_machine)])
    comp_res = {
        f"top_{k}_list": ranks,
    }
    return comp_res


def tok_k_overlap_far_j(
        human_data, machine_data,
        human_metadata=None, machine_metadata=None,
        symmetric=False,
        k=3, neighbors_to_exclude=2,
        columns_meaning: str = "line",
        model_name: str = 'Salesforce/codegen-16B-multi',
        model_folder: str = '/mnt/huggingface_models'):
    """Check how many top-k overlaps.

    The neighbors_to_exclude decides how many neighboring lines to remove
    from the comparisosn.
    e.g. neighbors_to_exclude=2, columns_meaning='line'  means that the
    current line of the starting token and the two lines above and below are
    removed from the ranking comparisons.

    columns_meaning: str
        The meaning of a column of human data or machine data.
        "line" means that the column represents a target line.
        "token" means that the column represents a target token.
    """
    assert human_data.shape == machine_data.shape
    n_matrix_lines = human_data.shape[0]

    if machine_metadata is None:
        parsed_tokenization = get_tokenization_from_human_metadata(
            human_metadata, model_name=model_name, model_folder=model_folder)
    else:
        parsed_tokenization = get_tokenization_from_machine_metadata(
            machine_metadata)
    raw_text = parsed_tokenization["raw_text"]
    tokens_prompt = parsed_tokenization["clean_tokens_prompt"]
    tokenization = parsed_tokenization["tokenization"]
    # print("raw_text", raw_text)
    # print("tokens_prompt", tokens_prompt)
    # print(tokenization[:20])

    ranks = []
    # print(f"length tokenization: {len(tokenization)}")
    # print(f"last tokens: {tokenization[-10:]}")
    # print(f"lines of matrix: {n_matrix_lines}")
    for i in range(n_matrix_lines):
        line_human_data = human_data[i]
        if np.sum(line_human_data) == 0:
            ranks.append([-1, i])
            continue
        line_machine_data = machine_data[i]
        if symmetric and np.sum(line_machine_data) == 0:
            ranks.append([-1, i])
            continue
        # get line and token position of current token i
        assert min([t['i'] for t in tokenization]) == 0, "tokenization is not 0-based"
        assert min([t['l'] for t in tokenization]) == 0, "line numbering does not start at 0"
        current_line_of_ith_token = tokenization[i]["l"]
        current_token_of_ith_line = tokenization[i]["i"]
        col_before = len(line_human_data)
        if columns_meaning == "token":
            # exclude in the ranking the current token and the tokens above and
            # below it as given by neighbors_to_exclude
            positions_to_exclude = list(range(
                current_line_of_ith_token - neighbors_to_exclude,
                current_line_of_ith_token + neighbors_to_exclude + 1))
            line_human_data = np.delete(line_human_data, positions_to_exclude)
            line_machine_data = np.delete(line_machine_data, positions_to_exclude)
        elif columns_meaning == "line":
            # same for lines
            positions_to_exclude = list(range(
                current_token_of_ith_line - neighbors_to_exclude,
                current_token_of_ith_line + neighbors_to_exclude + 1))
            all_line_indices = set([t["l"] for t in tokenization])
            val_and_line_human = list(zip(list(line_human_data), all_line_indices))
            line_human_data = np.array([
                v for v, l in val_and_line_human
                if l not in positions_to_exclude])
            val_and_line_machine = list(zip(list(line_machine_data), all_line_indices))
            line_machine_data = np.array([
                v for v, l in val_and_line_machine
                if l not in positions_to_exclude])
        col_after = len(line_human_data)
        #print(f"line {i}: {col_before} columns -> {col_after} columns")

        ranking_human = line_human_data.argsort()[::-1]
        ranking_machine = line_machine_data.argsort()[::-1]
        top_k_human = ranking_human[:k]
        top_k_machine = ranking_machine[:k]
        top_k_overlap = len(set(top_k_human) & set(top_k_machine))
        ranks.append([top_k_overlap, i])
    comp_res = {
        f"top_{k}_list": ranks,
    }
    return comp_res


def top_10_overlap(human_data, machine_data, symmetric=False):
    """Check how many top-10 overlaps."""
    return top_k_overlap(human_data, machine_data, symmetric, k=10)


def top_3_overlap(human_data, machine_data, symmetric=False):
    """Check how many top-3 overlaps."""
    return top_k_overlap(human_data, machine_data, symmetric, k=3)


def mse(human_data, machine_data):
    """Calculate the mean squared error between the two data."""
    assert human_data.shape == machine_data.shape
    res = np.mean((human_data - machine_data) ** 2)
    comp_res = {
        "msed": res,
    }
    return comp_res


def mse_line_by_line(human_data, machine_data, symmetric=False):
    """Calculate the MSE between the two data line by line."""
    assert human_data.shape == machine_data.shape
    n_lines = human_data.shape[0]
    mse_list = []
    for i in range(n_lines):
        line_human_data = human_data[i, :]
        if np.sum(line_human_data) == 0:
            mse_list.append([-1, i])
            continue
        line_machine_data = machine_data[i, :]
        if symmetric and np.sum(line_machine_data) == 0:
            mse_list.append([-1, i])
            continue
        mse_line = np.mean((line_human_data - line_machine_data) ** 2)
        mse_list.append([mse_line, i])
    comp_res = {
        "mse_list": mse_list,
    }
    return comp_res


def jsd_line_by_line(human_data, machine_data):
    """Calculate the JSD between the two data line by line."""
    assert human_data.shape == machine_data.shape
    n_lines = human_data.shape[0]
    jsd_list = []
    for i in range(n_lines):
        line_human_data = human_data[i, :]
        line_machine_data = machine_data[i, :]
        jsd_line = jensenshannon(line_human_data, line_machine_data)
        if np.isnan(jsd_line):
            jsd_line = None
        jsd_list.append(jsd_line)
    comp_res = {
        "jsd_list": jsd_list,
    }
    return comp_res


def spearman_rank_by_line(human_data, machine_data, symmetric=False):
    """Calculate the spearman rank correlation between the two data line by line."""
    assert human_data.shape == machine_data.shape
    n_lines = human_data.shape[0]
    spearman_correlation_list = []
    spearman_pavlaue_list = []
    i_list = []
    for i in range(n_lines):
        i_list.append(i)
        line_human_data = human_data[i, :]
        if np.sum(line_human_data) == 0:
            spearman_correlation_list.append(-1)
            spearman_pavlaue_list.append(-1)
            continue
        line_machine_data = machine_data[i, :]
        if symmetric and np.sum(line_machine_data) == 0:
            spearman_correlation_list.append(-1)
            spearman_pavlaue_list.append(-1)
            continue
        spearman_line = spearmanr(line_human_data, line_machine_data)
        if np.isnan(spearman_line.correlation):
            spearman_correlation_list.append(None)
            spearman_pavlaue_list.append(None)
        else:
            spearman_correlation_list.append(spearman_line.correlation)
            spearman_pavlaue_list.append(spearman_line.pvalue)
    comp_res = {
        "spearman_res_list": list(zip(
            spearman_correlation_list,
            spearman_pavlaue_list,
            i_list)),
    }
    return comp_res
