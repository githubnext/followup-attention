"""Utility to convert node distance to attention weights.

Note that all these functions get the same input:
    - distance: distance of the current node from the relevant variable node
    - max_distance: maximum distance from the relevant variable node in the
        current program
    - n_lines: number of lines in the current program
"""

def convert_distance_to_weight_naive(
        distance: int, max_distance: int, n_lines: int) -> float:
    """Convert a distance to an attention based distance.

    If the distance is infinite (i.e. the node is not reachable from the
    relevant variable node), then it get distance=None.
    """
    if distance is None:
        return 0
    att_weight = 1 - (distance / max_distance)
    return att_weight
