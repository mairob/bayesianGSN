from enum import Enum


class EGsnType(Enum):
    """Enumeration of GSN-Tree element types."""

    # 'Core' elements as defined in the standard (see also Goal Structuring Notation Community Standard Version 3 -- Section 1.2)
    GOAL = "goal"  ## in Nesic et. al 2021 this is a: claim (see Nesci et al. 2021, Definition 5)
    CONTEXT = "context"  ## in Nesic et. al 2021 this is a: claim (see Nesci et al. 2021, Definition 5)
    SOLUTION = "solution"  ## in Nesic et. al 2021 this is a: evidence (see Nesci et al. 2021, Definition 8)
    ASSUMPTION = "assumption"  ## in Nesic et. al 2021 this is a: axiom
    STRATEGY = "strategy"  ## in Nesic et. al 2021 this is a: inference rule (see Nesci et al. 2021, Definition 7)
    JUSTIFICATION = "justification"  ## in Nesic et. al 2021 this is a: axiom


class EGateModel(Enum):
    """Enumeration of supported types for boolean-like (canonical) aggregation gates.
    For technical information see Diez & Druzdel, 2007, https://www.cisiad.uned.es/techreports/canonical.pdf.

    The term 'noisy' refers to the possibility that SOME of the causes
    fail to produce the effect even when they are present.

    The term 'leaky' describes cases in which an inhibitor (leak) with probability qL
    may prevent the occurrence of Y even when ALL the conditions explicit in the model are fullled.
    """

    AND = "and"  # default aggregation type
    LEAKY_AND = "leaky_and"
    NOISY_AND = "noisy_and"
    # SIMPLE_AND = "simple_and"
    OR = "or"
    LEAKY_OR = "leaky_or"
    NOISY_OR = "noisy_or"
