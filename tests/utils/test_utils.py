from unittest.mock import MagicMock, patch

import pytest

from bayesiangsn.utils.Utils import is_float, is_valid_prob


@pytest.mark.parametrize(
    "raw_string, expected",
    [
        ("float", False),
        ("5", True),
        ("1.234", True),
        ("inf", True),
        ("one.3", False),
        ("one", False),
    ],
)
def test_is_float(raw_string, expected):
    assert is_float(raw_string) == expected


@pytest.mark.parametrize(
    "prob_vals, expected",
    [
        ([0.0, 0.1, 1.0], True),
        ([0.9, 9.0], False),
        ([[0.9, 1.0], [1.0, 0.9]], True),
        ([1.0], True),
        (0.0, True),
        (-1e-20, False),
        ([1, 0], True),
    ],
)
def test_is_valid_prob(prob_vals, expected):
    assert is_valid_prob(prob_vals) == expected
