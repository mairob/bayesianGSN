from typing import Dict, List, Optional, Tuple

import networkx as nx
import yaml
from networkx.classes.digraph import DiGraph

from bayesiangsn.core.Enums import EGsnType
from bayesiangsn.core.GsnElement import GsnElement


class GsnTree:
    """Main class for managing a parsed Goal Structuring Notation tree as a DiGraph.

    Attributes:
        tree_elements (Dict<str, GsnElement>): Elements (i.e., nodes) of the GSN tree
        name (str): Name of this GSN tree
        node_connections (list<tuple<str, str>>): List of tuples defining the edges between tree elements.
        tree_obj (networkx.DiGraph): Parsed GSN tree as real tree structure. Nodes.data contain objects of type GsnElement.
    """

    tree_elements = None
    name = None
    root = None
    node_connections = None
    tree_obj = None

    def __init__(self, name: str, yaml_path: Optional[str] = None) -> None:
        """Ctor of the GsnTree class.

        Args:
            name (str): Name of this GSN tree instance of type GsnElement.
        """
        self.name = name
        if yaml_path:
            self.load_gsn(yaml_path)

    def load_gsn(self, yaml_path: str) -> DiGraph:
        self.tree_elements = self._parse_yaml(yaml_path)
        self._verify_relations_valid(self.tree_elements)
        self.node_connections = self._parse_connections(self.tree_elements)
        self.tree_obj = self._create_tree(self.node_connections, self.tree_elements)

        return self.tree_obj

    def _parse_yaml(self, yaml_path: str) -> Dict:
        prefix_map_yaml = {
            "G": EGsnType.GOAL,
            "A": EGsnType.ASSUMPTION,
            "J": EGsnType.JUSTIFICATION,
            "Sn": EGsnType.SOLUTION,
            "C": EGsnType.CONTEXT,
            "S": EGsnType.STRATEGY,
        }

        tree_elements = {}
        with open(yaml_path) as file:
            data = yaml.safe_load(file)

            for node_name, vals in data.items():
                element_type = (
                    EGsnType.SOLUTION
                    if node_name.startswith("Sn")
                    else prefix_map_yaml.get(node_name[0], None)
                )

                if not element_type:
                    raise ValueError(
                        f"Parsed node with name {node_name} uses an undefined prefix {node_name[:2]}."
                    )
                gsn_element = GsnElement(
                    label=node_name,
                    intent=vals.get("text", None),
                    element_type=element_type,
                    motivation=None,
                    is_supported_by=vals.get("supportedBy", None),
                    in_context_of=vals.get("inContextOf", None),
                )

                # parse additional data:
                data = {}
                data["belief"] = vals.get("belief", None)
                gsn_element.data = data
                tree_elements[node_name] = gsn_element

        return tree_elements

    def _verify_relations_valid(self, tree_elements: Dict) -> None:
        # we need to check that the provided GSN tree has valid relationships
        # these are defined in Table 1:2-2 Core GSN Relationships in the GSN Community Standard Version 3 (page 18)
        # we use a fail-fast strategy, that means if one connection is invalid we stop.

        valid_support_relations = [
            (EGsnType.GOAL, EGsnType.GOAL),
            (EGsnType.GOAL, EGsnType.STRATEGY),
            (EGsnType.GOAL, EGsnType.SOLUTION),
            (EGsnType.STRATEGY, EGsnType.GOAL),
        ]

        valid_contex_relations = [
            (EGsnType.GOAL, EGsnType.CONTEXT),
            (EGsnType.GOAL, EGsnType.ASSUMPTION),
            (EGsnType.GOAL, EGsnType.JUSTIFICATION),
            (EGsnType.STRATEGY, EGsnType.CONTEXT),
            (EGsnType.STRATEGY, EGsnType.ASSUMPTION),
            (EGsnType.STRATEGY, EGsnType.JUSTIFICATION),
        ]

        for node in tree_elements.values():
            cur_support_relations = [
                (node.element_type, tree_elements[sup].element_type)
                for sup in node.supporters
            ]
            cur_context_relations = [
                (node.element_type, tree_elements[cntx].element_type)
                for cntx in node.contexts
            ]

            if not all([x in valid_support_relations for x in cur_support_relations]):
                raise ValueError(
                    f"Please check supportedBy arguments for node {node.label} if they define valid references."
                )
            if not all([x in valid_contex_relations for x in cur_context_relations]):
                raise ValueError(
                    f"Please check inContextOf arguments for node {node.label} if they define valid references."
                )

    def _parse_connections(self, tree_elements: Dict) -> List[str]:
        # we use the supportedBy // InContextOf information
        # these referenced nodes represent children/destination nodes in the GSN tree
        node_connections = []

        for label, element in tree_elements.items():
            for dest in element.supporters + element.contexts:
                connection = (label, dest)
                if connection not in node_connections:
                    node_connections.append(connection)

        return node_connections

    def _create_tree(
        self,
        node_connections: List[Tuple[str, str]],
        tree_elements: Dict[str, GsnElement],
    ) -> DiGraph:
        """Create a directed, acyclic graph from the GSN tree specification of the given gsn2x YAML-structure.

        Returns:
            networkx.DiGraph: Parsed tree structure as directed, acyclic graph object. Additional metadata is
                               added for each node via a custom attribute "data".

        Raises:
            ValueError: Raised if multiple roots should be encountered during the tree creation

        Args:
            node_connections ( list< tuple<src, dest>>): Node connections of the tree object.
            tree_elements (dict<str, GsnElement): Dictionary of parsed GSN Tree objects  with key: object str, value = GSN object instance
                specifying the data of a tree node.
        """
        model = nx.DiGraph()
        model.add_edges_from(node_connections)

        roots = list((v for v, d in model.in_degree() if d == 0))
        if len(roots) > 1:
            raise ValueError(
                f"Parsed GSN Tree might be invalid. Found multiple ({len(roots)} root nodes: { [tree_elements[root_id].label for root_id in roots] } "
            )

        self.root = roots[0]

        node_attributes = {
            node_id: {"data": gsn_object}
            for node_id, gsn_object in tree_elements.items()
        }

        nx.set_node_attributes(model, node_attributes)

        return model
