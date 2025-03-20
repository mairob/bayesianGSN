from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD as pgmpyCPT

matplotlib.use("Agg")


# def generateCondProbs(var_card, ev_cards):
#     length = np.prod(ev_cards) if len(ev_cards) > 0 else 1

#     res = []
#     for i in range(length):
#         cur_probs = np.abs(np.random.normal(size=var_card))
#         cur_probs /= cur_probs.sum()
#         res.append(cur_probs)

#     return np.array(res).transpose()


# def generate_states(name, nr_states):
#     return [f"st_{name}_{i}" for i in range(nr_states)]


# def generate_cpt_params(var_card, ev_cards):
#     if not isinstance(var_card, int):
#         raise TypeError(
#             f"Variable cardinality needs to be an integer but is {type(var_card)}"
#         )

#     if not isinstance(ev_cards, list):
#         raise TypeError(
#             f"Evidence cardinalities need to be provided as a list but are {type(ev_cards)}"
#         )

#     if var_card < 2:
#         raise ValueError(
#             f"Variable cardinality needs to be at least 2 but is {var_card}"
#         )

#     if not all([card >= 2 for card in ev_cards]):
#         raise ValueError(
#             f"Evidences need to be at least of cardinality 2 but are {ev_cards}"
#         )

#     alphabet = [chr(letter).upper() for letter in range(ord("a"), ord("z") + 1)]
#     var_name = alphabet.pop(0)
#     evidence = alphabet[: len(ev_cards)] if len(ev_cards) > 0 else []

#     state_names = {
#         var: generate_states(var, nr_states)
#         for var, nr_states in zip([var_name] + evidence, [var_card] + ev_cards)
#     }

#     values = generateCondProbs(var_card=var_card, ev_cards=ev_cards)

#     return var_name, evidence, state_names, values


# @pytest.fixture
# def fix_create_valid_cpt(request):
#     var_card = request.param["var_card"]
#     ev_cards = request.param["ev_cards"]

#     var_name, evidence, state_names, values = generate_cpt_params(var_card, ev_cards)

#     return CPT(
#         name=var_name,
#         variable_card=var_card,
#         values=values,
#         evidence=evidence,
#         evidence_card=ev_cards,
#         state_names=state_names,
#     )


# @pytest.fixture
# def fix_create_valid_pgmpy_cpt(request):
#     var_card = request.param["var_card"]
#     ev_cards = request.param["ev_cards"]

#     var_name, evidence, state_names, values = generate_cpt_params(var_card, ev_cards)

#     return pgmpyCPT(
#         variable=var_name,
#         variable_card=var_card,
#         values=values,
#         evidence=evidence,
#         evidence_card=ev_cards,
#         state_names=state_names,
#     )


# @pytest.fixture
# def fix_create_valid_pyagrum_cpt(request):
#     var_card = request.param["var_card"]
#     ev_cards = request.param["ev_cards"]

#     var_name, evidence, _, _ = generate_cpt_params(var_card, ev_cards)

#     pyagrum_cpt = gum.Potential()
#     pyagrum_cpt.add(gum.LabelizedVariable(var_name, var_name, int(var_card)))

#     for ev, card in zip(evidence, ev_cards):
#         pyagrum_cpt.add(gum.LabelizedVariable(ev, ev, int(card)))

#     pyagrum_cpt.randomCPT()
#     return pyagrum_cpt


# @pytest.fixture
# def fixture_bn_confounder_param():

#     def _make_bn(bn_name):
#         node_connections = [("NODE_A", "NODE_B"), ("NODE_A", "NODE_C")]
#         bn = BayesianNetwork(bn_name, node_connections)

#         NODE_A = CPT(
#             name="NODE_A",
#             variable_card=2,
#             values=[[0.1], [0.9]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]},
#         )
#         NODE_B = CPT(
#             name="NODE_B",
#             variable_card=2,
#             values=[[0.12, 0.34], [0.88, 0.66]],
#             evidence=["NODE_A"],
#             evidence_card=[2],
#             state_names={
#                 "NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"],
#                 "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"],
#             },
#         )
#         NODE_C = CPT(
#             name="NODE_C",
#             variable_card=2,
#             values=[[0.98, 0.76], [0.02, 0.24]],
#             evidence=["NODE_A"],
#             evidence_card=[2],
#             state_names={
#                 "NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"],
#                 "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"],
#             },
#         )

#         bn.add_cpts(NODE_A, NODE_B, NODE_C)

#         marginals = {
#             "NODE_A": [[0.1], [0.9]],
#             "NODE_B": [[0.318], [0.682]],
#             "NODE_C": [[0.782], [0.218]],
#         }

#         return bn, marginals

#     return _make_bn


# @pytest.fixture
# def fixture_bn_collider_param():

#     def _make_bn(bn_name):
#         node_connections = [("NODE_A", "NODE_C"), ("NODE_B", "NODE_C")]
#         bn = BayesianNetwork(bn_name, node_connections)

#         NODE_A = CPT(
#             name="NODE_A",
#             variable_card=2,
#             values=[[0.123], [0.877]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]},
#         )
#         NODE_B = CPT(
#             name="NODE_B",
#             variable_card=2,
#             values=[[0.987], [0.013]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]},
#         )
#         NODE_C = CPT(
#             name="NODE_C",
#             variable_card=2,
#             values=[[0.123, 0.456, 0.789, 0.987], [0.877, 0.544, 0.211, 0.013]],
#             evidence=["NODE_A", "NODE_B"],
#             evidence_card=[2, 2],
#             state_names={
#                 "NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"],
#                 "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"],
#                 "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"],
#             },
#         )

#         bn.add_cpts(NODE_A, NODE_B, NODE_C)

#         marginals = {
#             "NODE_A": [[0.123], [0.877]],
#             "NODE_B": [[0.987], [0.013]],
#             "NODE_C": [[0.71], [0.29]],
#         }

#         return bn, marginals

#     return _make_bn


# @pytest.fixture
# def fixture_bn_independent_nodes_only_param():

#     def _make_bn(bn_name):
#         node_connections = []
#         bn = BayesianNetwork(bn_name, node_connections)

#         nodes = ["NODE_A", "NODE_B", "NODE_C"]
#         for node in nodes:
#             bn.add_node(node)

#         NODE_A = CPT(
#             name="NODE_A",
#             variable_card=2,
#             values=[[0.123], [0.877]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]},
#         )
#         NODE_B = CPT(
#             name="NODE_B",
#             variable_card=2,
#             values=[[0.987], [0.013]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]},
#         )
#         NODE_C = CPT(
#             name="NODE_C",
#             variable_card=2,
#             values=[[0.456], [0.544]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"]},
#         )

#         bn.add_cpts(NODE_A, NODE_B, NODE_C)

#         marginals = {
#             "NODE_A": [[0.123], [0.877]],
#             "NODE_B": [[0.987], [0.013]],
#             "NODE_C": [[0.456], [0.544]],
#         }

#         return bn, marginals

#     return _make_bn


# @pytest.fixture
# def fixture_bn_causal_queries_param():

#     def _make_bn(bn_name):
#         node_connections = [
#             ("NODE_A", "NODE_C"),
#             ("NODE_B", "NODE_D"),
#             ("NODE_C", "NODE_D"),
#         ]
#         bn = BayesianNetwork(bn_name, node_connections)

#         NODE_A = CPT(
#             name="NODE_A",
#             variable_card=2,
#             values=[[0.123], [0.877]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"]},
#         )
#         NODE_B = CPT(
#             name="NODE_B",
#             variable_card=2,
#             values=[[0.987], [0.013]],
#             evidence=None,
#             evidence_card=None,
#             state_names={"NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"]},
#         )

#         NODE_C = CPT(
#             name="NODE_C",
#             variable_card=2,
#             values=[[0.321, 0.654], [0.679, 0.346]],
#             evidence=["NODE_A"],
#             evidence_card=[2],
#             state_names={
#                 "NODE_A": ["STATE_NODE_A_Yes", "STATE_NODE_A_No"],
#                 "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"],
#             },
#         )

#         NODE_D = CPT(
#             name="NODE_D",
#             variable_card=2,
#             values=[[0.123, 0.456, 0.789, 0.987], [0.877, 0.544, 0.211, 0.013]],
#             evidence=["NODE_B", "NODE_C"],
#             evidence_card=[2, 2],
#             state_names={
#                 "NODE_B": ["STATE_NODE_B_Yes", "STATE_NODE_B_No"],
#                 "NODE_C": ["STATE_NODE_C_Yes", "STATE_NODE_C_No"],
#                 "NODE_D": ["STATE_NODE_D_Yes", "STATE_NODE_D_No"],
#             },
#         )

#         bn.add_cpts(NODE_A, NODE_B, NODE_C, NODE_D)

#         return bn

#     return _make_bn


# @pytest.fixture
# def fixture_bn_random_small():
#     """This method produces the following strcuture
#     A->C->D->E; B->D->E; A->E; B->E, C->E
#     with a random cardinality for each node
#     and valid, random CPT values for each node
#     for each instantiation.
#     The cardinality ranges are [2 ...7]
#     """

#     def _make_bn(bn_name):
#         node_connections = [
#             ("A", "C"),
#             ("B", "D"),
#             ("C", "D"),
#             ("A", "E"),
#             ("B", "E"),
#             ("C", "E"),
#             ("D", "E"),
#         ]
#         bn = BayesianNetwork(bn_name, node_connections)

#         rnd_variable_cardinalities = np.random.randint(low=2, high=8, size=5)
#         variable_names = ["A", "B", "C", "D", "E"]
#         state_names = {
#             name: generate_states(name, nr_states)
#             for name, nr_states in zip(variable_names, list(rnd_variable_cardinalities))
#         }

#         NODE_A = CPT(
#             name="A",
#             variable_card=rnd_variable_cardinalities[0],
#             values=generateCondProbs(
#                 var_card=rnd_variable_cardinalities[0], ev_cards=[]
#             ),
#             evidence=None,
#             evidence_card=None,
#             state_names={var: state_names[var] for var in ["A"] if var in state_names},
#         )
#         NODE_B = CPT(
#             name="B",
#             variable_card=rnd_variable_cardinalities[1],
#             values=generateCondProbs(
#                 var_card=rnd_variable_cardinalities[1], ev_cards=[]
#             ),
#             evidence=None,
#             evidence_card=None,
#             state_names={var: state_names[var] for var in ["B"] if var in state_names},
#         )

#         NODE_C = CPT(
#             name="C",
#             variable_card=rnd_variable_cardinalities[2],
#             values=generateCondProbs(
#                 var_card=rnd_variable_cardinalities[2],
#                 ev_cards=list([rnd_variable_cardinalities[0]]),
#             ),
#             evidence=["A"],
#             evidence_card=rnd_variable_cardinalities[0],
#             state_names={
#                 var: state_names[var] for var in ["A", "C"] if var in state_names
#             },
#         )

#         NODE_D = CPT(
#             name="D",
#             variable_card=rnd_variable_cardinalities[3],
#             values=generateCondProbs(
#                 var_card=rnd_variable_cardinalities[3],
#                 ev_cards=list(rnd_variable_cardinalities[[1, 2]]),
#             ),
#             evidence=["B", "C"],
#             evidence_card=list(rnd_variable_cardinalities[[1, 2]]),
#             state_names={
#                 var: state_names[var] for var in ["B", "C", "D"] if var in state_names
#             },
#         )

#         NODE_E = CPT(
#             name="E",
#             variable_card=rnd_variable_cardinalities[4],
#             values=generateCondProbs(
#                 var_card=rnd_variable_cardinalities[4],
#                 ev_cards=list(rnd_variable_cardinalities[:-1]),
#             ),
#             evidence=["A", "B", "C", "D"],
#             evidence_card=list(rnd_variable_cardinalities[:-1]),
#             state_names=state_names,
#         )

#         bn.add_cpts(NODE_A, NODE_B, NODE_C, NODE_D, NODE_E)

#         return bn

#     return _make_bn


# @pytest.fixture
# def fixture_patched_nestor_instance():
#     with patch("pynestor.core.ModelManager"), patch(
#         "pynestor.algorithms.NestorAlgorithmManager"
#     ):
#         return Nestor()
