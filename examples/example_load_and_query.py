import os
import sys

cur_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(cur_dir_path, os.pardir)))

from bayesiangsn.core.Enums import EGateModel
from bayesiangsn.core.GsnTree import GsnTree
from bayesiangsn.NesicGsnTree import NesicBayesianGsnTree

TEST_FILE_1 = r"bayesiangsn\data\example_nesic_eval_with_probs.yaml"
TEST_FILE_2 = r"bayesiangsn\data\example_nesic_eval_20Hazards_prob.yaml"

gsn_tree = GsnTree("Exmpl_1", TEST_FILE_1)

nesic_1 = NesicBayesianGsnTree("Exmpl_1", gsn_tree)
print(nesic_1.query_belief_in_goal())

# Check a second example
gsn_tree.load_gsn(TEST_FILE_2)
nesic_2 = NesicBayesianGsnTree("Exmpl_2", gsn_tree)

# Change aggregation type of a Gate -> introducing noise or leak
nesic_2.change_goal_aggregation(
    goal="G1",
    gate_model=EGateModel.LEAKY_OR,
    prob_values=[0.9, 0.99, 0.95, 0.99],
    leak=0.1,
)
print(nesic_2.query_belief_in_goal("G1"))
