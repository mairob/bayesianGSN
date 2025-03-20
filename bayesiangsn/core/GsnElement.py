from typing import Dict, List, Optional

from bayesiangsn.core.Enums import EGsnType


class GsnElement:
    """
    Instances represent nodes in a parsed GSN tree (DiGraph).
    See also 'core' GSN elements as defined by the Goal Structuring Notation Community Standard Version 3 -- Section 1.2.
    """

    def __init__(
        self,
        label: str,
        intent: str,
        element_type: EGsnType,
        motivation: Optional[str] = None,
        is_supported_by: Optional[List[str]] = None,
        in_context_of: Optional[List[str]] = None,
        data: Optional[Dict] = None,
    ) -> None:
        if not isinstance(label, str):
            raise TypeError(
                f"Label needs to be a string, but a {type(label)} was provided for {label}."
            )
        if not isinstance(intent, str):
            raise TypeError(
                f"Intent needs to be a string, but a {type(intent)} was provided for {label}."
            )

        self._label = label
        self._intent = intent
        self._element_type = element_type
        self._motivation = motivation if isinstance(motivation, str) else None
        self.supporters = is_supported_by
        self.contexts = in_context_of
        self.data = data

    @property
    def label(self) -> str:
        return self._label

    @property
    def intent(self) -> str:
        return self._intent

    @property
    def element_type(self) -> EGsnType:
        return self._element_type

    @property
    def motivation(self) -> str:
        return self._motivation if self._motivation else "No motivation available."

    @property
    def supporters(self) -> List[str]:
        return self._is_supported_by

    @supporters.setter
    def supporters(self, supporters: List[str]) -> None:
        if supporters and not all(isinstance(x, str) for x in supporters):
            raise TypeError(
                f"Each element in 'supporters' must be a string for {self.label}."
            )
        self._is_supported_by = supporters if supporters else []

    @property
    def contexts(self) -> List[str]:
        return self._in_context_of

    @contexts.setter
    def contexts(self, contexts: List[str]) -> None:
        if contexts and not all(isinstance(x, str) for x in contexts):
            raise TypeError(
                f"Each element in 'contexts' must be a string for {self.label}."
            )
        self._in_context_of = contexts if contexts else []

    @property
    def data(self) -> Dict:
        return self._data

    @data.setter
    def data(self, data: Dict) -> None:
        if data and not isinstance(data, dict):
            raise TypeError(f"Additional data for {self.label} must be a dictionary.")
        self._data = data if data else {}
