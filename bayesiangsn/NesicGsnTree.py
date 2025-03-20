import copy
from typing import Dict, List, Optional, Tuple, Union

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

from bayesiangsn.core.CanonicalCPT import create_binary_logic_gate
from bayesiangsn.core.Enums import EGateModel, EGsnType
from bayesiangsn.core.GsnElement import GsnElement
from bayesiangsn.core.GsnTree import GsnTree
from bayesiangsn.utils.Utils import is_valid_prob


class NesicBayesianGsnTree:
    """Main class for managing a Goal Structuring Notation tree as a Bayesian Network.
       The transformation logic is given by Nesic et al. 2021 (https://doi.org/10.1016/j.ssci.2021.105187).

    Attributes:
        tree_elements (Dict<str, GsnElement>): Elements (i.e., nodes) of the raw/parsed GSN tree
        name (str): Name of this BN-based GSN tree
        node_connections (list<tuple<str, str>>): List of tuples defining the edges between tree elements in the BN representation.
        tree_obj (networkx.DiGraph): Parsed GSN tree as real tree structure. Nodes.data contain objects of type GsnElement.
    """

    def __init__(self, name: str, gsn_tree: GsnTree) -> None:
        """Ctor of the NesicBayesianGsnTree class implementing a BN according to Nesic et al. 2021 (https://doi.org/10.1016/j.ssci.2021.105187)

        Args:
            name (str): Name of this GSN tree.
            gsn_tree(GsnTree): Original networkX-DiGraph instance of the GSN-Tree (e.g. loaded from a gsn2X YAML)
        """
        self._name = name
        self._gsn_tree = gsn_tree

        self._check_completeness_of_argument(self._gsn_tree)
        self._check_well_formdness(self._gsn_tree)
        self._gurantee_inference_rules(gsn_tree)
        self._bn = self._create_bn(self._gsn_tree)

        if self._implicit_inf_rules:
            print("The following implict inference rules (Solutions) were added:")
            for impl_rule in self._implicit_inf_rules.keys():
                print(f"{impl_rule}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def gsn_tree(self) -> GsnTree:
        return self._gsn_tree

    @property
    def bn(self) -> BayesianNetwork:
        return self._bn

    @property
    def implict_rules(self) -> Dict[str, GsnElement]:
        return self._implicit_inf_rules

    def _check_well_formdness(self, gsn_tree):
        """Check the neccessary constraints on a GSN tree as outlined in Defintion 11 of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187)
        to ensure it is a well-formed GSN argumentation.
        """

        # i) Nodes of type goal cannot connect to other nodes of type goal
        if any(
            [
                EGsnType.GOAL
                == self._gsn_tree.tree_elements[src].element_type
                == self._gsn_tree.tree_elements[dest].element_type
                for src, dest in self._gsn_tree.node_connections
            ]
        ):
            raise ValueError(
                f"Well-formdness constraint i) violated: Nodes of type goal cannot connect to other nodes of type goal."
            )

        # ii) Each node of type goal connects either to exactly one node of type strategy, or to at least one type solution
        for label, node in self._gsn_tree.tree_elements.items():
            if node.element_type == EGsnType.GOAL:
                cnt_strategy_connections = sum(
                    [
                        (
                            1
                            if self._gsn_tree.tree_elements[x].element_type
                            == EGsnType.STRATEGY
                            else 0
                        )
                        for x in node.supporters
                    ]
                )

                if cnt_strategy_connections > 1:
                    raise ValueError(
                        f"Well-formdness constraint ii) violated: Each node of type goal connects to exactly one node of type strategy.\nViolated by node: {label}."
                    )

                elif cnt_strategy_connections == 1:
                    continue

                else:
                    cnt_solution_connections = sum(
                        [
                            (
                                1
                                if self._gsn_tree.tree_elements[x].element_type
                                == EGsnType.SOLUTION
                                else 0
                            )
                            for x in node.supporters
                        ]
                    )

                    if cnt_solution_connections >= 1:
                        continue
                    else:
                        raise ValueError(
                            f"Well-formdness constraint ii) violated: Each node of type goal connects to at least one type solution.\nViolated by node: {label}."
                        )

        # iii) Each node of type strategy connects to a node of type justification
        for label, node in self._gsn_tree.tree_elements.items():
            if node.element_type == EGsnType.STRATEGY:
                cnt_justification_connections = sum(
                    [
                        (
                            1
                            if self._gsn_tree.tree_elements[x].element_type
                            == EGsnType.JUSTIFICATION
                            else 0
                        )
                        for x in node.contexts
                    ]
                )

                if cnt_justification_connections < 1:
                    raise ValueError(
                        f"Well-formdness constraint iii) violated: Each node of type strategy connects to a node of type justification.\nViolated by node: {label}."
                    )

    def _check_completeness_of_argument(self, gsn_tree):
        """Check the existence of "implicit premises" as outlined in Defintion 12 of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187)
        Each goal is required a context, either directly or indirectly via the associated strategy
        """

        for label, node in gsn_tree.tree_elements.items():
            if node.element_type == EGsnType.GOAL:
                cnt_goal_justification_connections = sum(
                    [
                        (
                            1
                            if gsn_tree.tree_elements[x].element_type
                            == EGsnType.JUSTIFICATION
                            else 0
                        )
                        for x in node.contexts
                    ]
                )

                if cnt_goal_justification_connections < 1:
                    # the goal itself does not have a justification therefore the connected strategy must have one
                    for goal_context in node.contexts:
                        if (
                            gsn_tree.tree_elements[goal_context].element_type
                            == EGsnType.STRATEGY
                        ):
                            cnt_strategy_justification_connections = sum(
                                [
                                    (
                                        1
                                        if gsn_tree.tree_elements[x].element_type
                                        == EGsnType.JUSTIFICATION
                                        else 0
                                    )
                                    for x in gsn_tree.tree_elements[
                                        goal_context
                                    ].contexts
                                ]
                            )

                            if cnt_strategy_justification_connections < 1:
                                raise ValueError(
                                    f"Completeness constraint violated: Each node of type goal connects to a node of type context (or via its strategy).\nViolated by node: {label}."
                                )

    def _create_bn(self, gsn_tree):
        """Main logic to convert a well-formed GSN tree (according to Nesic et al.) into a BN representation"""
        # ToDo: Deal with "Assumptions" as they are expected to be always true --> therefore they can be ommitted from the BN'?
        cpts = {}
        axiom_types = [
            EGsnType.CONTEXT,
            EGsnType.JUSTIFICATION,
            EGsnType.ASSUMPTION,
        ]

        evidence_types = [EGsnType.SOLUTION]
        root_node_types = evidence_types + axiom_types

        # 1) make sure every "Goal" node has a "Strategy" node as a parent in the GSN tree (i.e., add implict inference rules X_psy if needed)
        mod_gsn_tree, bn_node_connections, implicit_inf_rules = (
            self._gurantee_inference_rules(gsn_tree)
        )
        self._implicit_inf_rules = implicit_inf_rules

        # 2) map according to Table 3 of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187)
        # Solutions --> root nodes X_e  || states: sat, notSat
        # Context, Justification, Assumption --> root nodes X_a  || states: sat, notSat

        # Strategy --> CPTs X_psy || states: sound, notSound || parents: X_a
        # Goals --> CPTs X_p  || states: sat, notSat || depending on GSN struct
        # MAIN Goal --> CPT X_q || states: sat, notSat || parents X_psy, X_p, Xa  --> represents logical AND for all parents = sat

        # 2.1) create root nodes X_e and X_a by direct transformation
        #      according to Table 3 of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187)
        #      Per Definition in Table 3: P(X_a=sat) = 1 || P(X_e=sat) = 1
        for label, node in gsn_tree.tree_elements.items():

            if node.element_type in root_node_types:
                prob_axiom_sat = (
                    node.data.get("belief", None)
                    if node.data.get("belief", None)
                    else 1.0
                )

                cpts[label] = TabularCPD(
                    variable=label,
                    variable_card=2,
                    values=[[prob_axiom_sat], [1 - prob_axiom_sat]],
                    evidence=None,
                    evidence_card=None,
                    state_names={label: ["sat", "notSat"]},
                )

        # 2.2) focus on explicit Strategy nodes X_psy, as they only have root nodes X_a (contexts) as direct parents
        #      which are given by the GSN design via to the completeness constraints
        #      The CPT values need to be MANUALLY set according to Type II CPT values (see Sec. 6.3.2, of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187))
        #      The CPT values care about P(sound | evidences are satisfied) --> for now we model this with a BOOLEAN AND
        for label, node in gsn_tree.tree_elements.items():
            if (
                node.element_type == EGsnType.STRATEGY
                and label not in self._implicit_inf_rules.keys()
            ):

                cur_state_names = {label: ["sound", "notSound"]}
                for k in node.contexts:
                    cur_state_names[k] = ["sat", "notSat"]

                cpts[label] = TabularCPD(
                    variable=label,
                    variable_card=2,
                    values=create_binary_logic_gate(
                        evidences=node.contexts, gate_model=EGateModel.AND
                    ),
                    evidence=node.contexts,
                    evidence_card=[2] * len(node.contexts),
                    state_names=cur_state_names,
                )

                bn_node_connections = bn_node_connections + [
                    (influence, label) for influence in node.contexts
                ]

        # 2.3) implicit Strategy nodes X_psy represent root nodes as they are artifically added
        #      The CPT values need to be MANUALLY set according to Type III CPT values (see Sec. 6.3.3, of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187))
        #      For now we fix all implicit beliefs to a predefined value

        for label, node in self._implicit_inf_rules.items():
            prob_implrule_sound = (
                node.data.get("belief", None) if node.data.get("belief", None) else 1.0
            )

            cpts[label] = TabularCPD(
                variable=label,
                variable_card=2,
                values=[
                    [prob_implrule_sound],
                    [1 - prob_implrule_sound],
                ],
                evidence=None,
                evidence_card=None,
                state_names={label: ["sound", "notSound"]},
            )

        # 2.4) X_p have as parents directly attached axioms X_a, (implicit) inference rules X_Psy, and by an associated Strategy as proxy, preceding premises/goals X_p
        #      Due to that indirect dependence on preceding goals X_p we need to create these nodes "recursively" as stated by Nesic et al.
        #      Preceding goals are by the "well-formedness constraints" parents of predecessor nodes of type Strategy in the GSN tree.
        #      The CPTs for X_p are set according to Type I CPT values and represent BOOLEAN ANDs (see Sec. 6.3.2, of Nesic et al., 2021 (https://doi.org/10.1016/j.ssci.2021.105187))
        for label, node in mod_gsn_tree.tree_elements.items():

            if node.element_type == EGsnType.GOAL:
                all_directly_related_nodes = [
                    x for x in mod_gsn_tree.tree_obj.successors(label)
                ]

                preceding_goals = []
                for x in all_directly_related_nodes:
                    if mod_gsn_tree.tree_elements[x].element_type == EGsnType.STRATEGY:
                        for y in mod_gsn_tree.tree_obj.successors(x):
                            if (
                                mod_gsn_tree.tree_elements[y].element_type
                                == EGsnType.GOAL
                            ):
                                preceding_goals.append(y)
                scoped_influences = all_directly_related_nodes + preceding_goals

                cur_state_names = {label: ["sat", "notSat"]}
                for k in scoped_influences:
                    cur_state_names[k] = (
                        ["sat", "notSat"]
                        if mod_gsn_tree.tree_elements[k].element_type
                        in root_node_types + [EGsnType.GOAL]
                        else ["sound", "notSound"]
                    )

                cpts[label] = TabularCPD(
                    variable=label,
                    variable_card=2,
                    values=create_binary_logic_gate(
                        evidences=scoped_influences, gate_model=EGateModel.AND
                    ),
                    evidence=scoped_influences,
                    evidence_card=[2] * len(scoped_influences),
                    state_names=cur_state_names,
                )

                bn_node_connections = bn_node_connections + [
                    (influence, label) for influence in scoped_influences
                ]

        model = BayesianNetwork(bn_node_connections)
        model.add_cpds(*list(cpts.values()))
        model.check_model()

        return model

    def _gurantee_inference_rules(self, gsn_tree):
        """Make sure every "Goal" node has a "Strategy" node as a parent in the GSN tree (i.e., add implict inference rules X_psy if needed)"""
        mod_gsn_tree = copy.deepcopy(gsn_tree)
        bn_node_connections = []
        implicit_inf_rules = {}

        for label, node in gsn_tree.tree_elements.items():
            if node.element_type == EGsnType.GOAL:
                if not any(
                    [
                        gsn_tree.tree_elements[x].element_type == EGsnType.STRATEGY
                        for x in node.supporters
                    ]
                ):
                    new_inf_rule = GsnElement(
                        label=f"implicit_S_{label}",
                        intent=f"Represents an added implicit inference rule for the goal: {label}",
                        element_type=EGsnType.STRATEGY,
                        motivation="Added due to BN constuction rules",
                        is_supported_by=None,
                        in_context_of=None,
                    )

                    # store for easier acces later on
                    implicit_inf_rules[new_inf_rule.label] = new_inf_rule

                    # also update the GSN tree
                    mod_gsn_tree.node_connections.append((new_inf_rule.label, label))
                    mod_gsn_tree.tree_obj.add_edge(label, new_inf_rule.label)
                    mod_gsn_tree.tree_obj.nodes[new_inf_rule.label].update(
                        {"data": new_inf_rule}
                    )
                    mod_gsn_tree.tree_elements[new_inf_rule.label] = new_inf_rule
                    bn_node_connections.append((new_inf_rule.label, label))

        return mod_gsn_tree, bn_node_connections, implicit_inf_rules

    def set_implict_beliefs(
        self, beliefs: Union[Tuple[str, float], Dict[str, float]]
    ) -> None:
        """Set the belief of implicit rules (i.e. Strategies that have been created to support a valid BN-based GSN tree)"""

        if not isinstance(beliefs, (tuple, dict)):
            raise TypeError(
                f"Provided beliefs need to be of type tuple<str, float> or dict<str, float> but are: {type(beliefs)}"
            )

        beliefs = {beliefs[0]: beliefs[1]} if isinstance(beliefs, tuple) else beliefs

        for node, val in beliefs.items():
            if node not in self._implicit_inf_rules.keys():
                raise ValueError(
                    f"Scoped element {node} is not an implicit inference rule or not part of the GSN tree."
                )

            if not is_valid_prob(val):
                raise ValueError(
                    f"Belief for element {node} needs to be between 0...1 but is {val}."
                )

            old_cpt = self._bn.get_cpds(node)
            self._bn.add_cpds(
                TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[val], [1 - val]],
                    evidence=None,
                    evidence_card=None,
                    state_names=old_cpt.state_names.copy(),
                )
            )

    def change_goal_aggregation(
        self,
        goal,
        gate_model: Union[str, EGateModel] = EGateModel.AND,
        prob_values: Optional[List[float]] = None,
        substitute_probs: Optional[List[float]] = None,
        leak: Optional[float] = None,
    ) -> None:
        """Change the default aggregation behaviour (AND) of a goal in the transformed GSN tree (i.e. the BN representation).
        Note: due to the intention of a GSN tree, aggregations should be AND-like."""

        old_cpt = self._bn.get_cpds(goal)
        self._bn.add_cpds(
            TabularCPD(
                variable=goal,
                variable_card=2,
                values=create_binary_logic_gate(
                    evidences=old_cpt.variables[1:],
                    gate_model=gate_model,
                    prob_values=prob_values,
                    substitute_probs=substitute_probs,
                    leak=leak,
                ),
                evidence=old_cpt.variables[1:],
                evidence_card=old_cpt.cardinality[1:],
                state_names=old_cpt.state_names.copy(),
            )
        )

    def query_belief_in_goal(
        self, goal: Optional[str] = None, evidence: Optional[Dict[str, str]] = None
    ) -> float:
        """Calculate the belief in a provided goal.
        If no arguments are provided, the belief in the main goal is caluclated
        """
        if goal:
            goal_node = self.gsn_tree.tree_elements.get(goal, None)
            if not goal_node:
                raise ValueError(
                    f"Provided goal ({goal}) is not part of the GSN tree scoped by this instance."
                )

            if goal_node.element_type != EGsnType.GOAL:
                raise ValueError(
                    f"Provided goal ({goal}) is of type {goal_node.element_type} but must be a 'Goal'"
                )
        else:
            goal = [n for n, d in self._gsn_tree.tree_obj.in_degree() if d == 0][0]
            print(f"Running calculation for primary goal: {goal}")

        infer = VariableElimination(self._bn)
        return infer.query([goal], evidence=evidence)
