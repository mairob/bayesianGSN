# bayesianGSN

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Overview
**bayesianGSN** is a Bayesian Network-based implementation to calculate the belief of a claim given via a Goal Structuring Notation (GSN) tree.

## STATUS
Currently this is an initial, untested, undocumented, and highly volatile proof-of-concept version.
Changes in the interfaces, directory layout, and so on are highly likely to occur.


## Installation
Ensure you are using **Python 3.10 or above**.

```bash
git clone https://github.com/mairob/bayesianGSN.git
cd bayesianGSN
python setup.py install
```

## Usage
```bash
TEST_FILE_1 = r"bayesiangsn\data\example_nesic_eval_with_probs.yaml"
TEST_FILE_2 = r"bayesiangsn\data\example_nesic_eval_20Hazards_prob.yaml"

# Load a GSN from a gsn2x - style YAML
gsn_tree = GsnTree("Exmpl_1", TEST_FILE_1)

# Transform it into a Bayesian Network and evaluate the belief in a Goal
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
```

## Examples
The "examples/" directory contains simple API examples on how to use this packages features:
- **example_load_and_query.py**: Demonstrates the evaluation of a GSN tree loaded from a YAML file (see also [gsn2x](https://jonasthewolf.github.io/gsn2x/) for the file format)


## Author
- Robert Maier - [LinkedIn](https://www.linkedin.com/in/robert-maier-ete-dl/)
