from itertools import product
from typing import List, Optional, Union

import numpy as np

from bayesiangsn.core.Enums import EGateModel
from bayesiangsn.utils.Utils import is_float, is_valid_prob


def create_binary_logic_gate(
    evidences: List[str],
    gate_model: Union[str, EGateModel],
    prob_values: Optional[List[float]] = None,
    substitute_probs: Optional[List[float]] = None,
    leak: Optional[float] = None,
) -> List[List[float]]:
    """
    Create a canonical CPT based on boolean logic gates.
    See Diez & Druzdel, 2007, https://www.cisiad.uned.es/techreports/canonical.pdf.

    prob_values: A list of probabilities values for each `evidence` variable to trigger e.g. inhibit a noisy gate.
                Provided values e.g. for a NoisyOR represent P(y=False | x_i = True, Z=False) for all other parents variables Z.
                They therfore represent the likelihood that an effect is NOT realized even tho a valid trigger x_i is present.
    """
    if prob_values:
        prob_values = (
            np.array([prob_values])
            if not isinstance(prob_values, list)
            else np.array(prob_values)
        )

        if not is_valid_prob(prob_values):
            raise ValueError(
                f"Provided probabilities need to be between 0...1 but are {prob_values}."
            )

    # relevant for leaky|noisy AND
    if substitute_probs:
        if not len(substitute_probs) == len(evidences):
            raise ValueError(
                f"Substitute probabilities need to be provided for all parental nodes of a noisy|leaky AND."
            )

        substitute_probs = np.array(substitute_probs)

        if not is_valid_prob(substitute_probs):
            raise ValueError(
                f"Provided substitute probabilities need to be between 0...1 but are {substitute_probs}."
            )

    if leak and not is_valid_prob(leak):
        raise ValueError(f"Leak probability needs to be between 0...1")

    if len(evidences) < 0 or len(evidences) > 31:
        # 31 is due to the maximum supported number of parents in pgmpy
        raise ValueError(f"Number of binary evidences is out of bounds (0...31).")

    if isinstance(gate_model, str):
        gate_model = gate_model.lower()

    # corresponds to P(y+|parent state combination)
    gate_vals = np.zeros(2 ** len(evidences))

    # corresponds to the I_+ -selector function in  Diez & Druzdel, 2007, https://www.cisiad.uned.es/techreports/canonical.pdf
    # we assume the state order True | False for all binary nodes
    state_selectors = product([True, False], repeat=len(evidences))

    canonical_cpt_function = None

    match gate_model:
        case EGateModel.AND | "and":
            canonical_cpt_function = lambda s: 1.0 if all(s) else 0.0
        case EGateModel.OR | "or":
            canonical_cpt_function = lambda s: 1.0 if any(s) else 0.0
        case EGateModel.NOISY_AND | "noisy_and":
            canonical_cpt_function = lambda s: np.prod(
                1.0 - prob_values[np.array(s)]
            ) * np.prod(substitute_probs[np.invert(np.array(s))])
        case EGateModel.LEAKY_AND | "leaky_and":
            canonical_cpt_function = (
                lambda s: (1 - leak)
                * np.prod(1.0 - prob_values[np.array(s)])
                * np.prod(substitute_probs[np.invert(np.array(s))])
            )
        # case EGateModel.SIMPLE_AND | "simple_and":
        #     canonical_cpt_function = lambda s: sat_prob if all(s) else fail_prob
        case EGateModel.NOISY_OR | "noisy_or":
            canonical_cpt_function = lambda s: 1.0 - np.prod(prob_values[np.array(s)])
        case EGateModel.LEAKY_OR | "leaky_or":
            canonical_cpt_function = lambda s: 1.0 - (1.0 - leak) * np.prod(
                prob_values[np.array(s)]
            )
        case _:
            raise TypeError(f"Unsupported gate type: {gate_model}")

    # Compute gate values using pre-selected function
    for i, state_combination in enumerate(state_selectors):
        gate_vals[i] = canonical_cpt_function(state_combination)

    return np.stack((gate_vals, 1.0 - gate_vals))
