from typing import List, Union

import numpy as np


def is_float(x: str) -> bool:
    """Checks if a given string represents a float value.

    Args:
        x (str): Provided float-value candidate.

    Returns:
        bool: True if given string can be converted into a float variable
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def is_valid_prob(x: Union[float, List[float]]) -> bool:
    """Checks if a given float or array of floats contains valid probability values (0...1).

    Args:
        x (Union[float, List[float]]): Provided (array of) probability values.

    Returns:
        bool: True if (all) given probability values are between 0 and 1.
    """
    x = np.array(x).flatten()
    return False if any(x > 1) or any(x < 0) else True
