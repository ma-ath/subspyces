from typing import List, Union

from .vector_space import VectorSpace


class VectorSet:
    """
    Class that defines a set of vector spaces
    """
    def __init__(self, subspaces: List[VectorSpace] = []) -> None:
        """
        subspaces: list of VectorSpace
        """
        self._set = {}
        self.append(subspaces)

    def append(self, subspaces: Union[VectorSpace, List[VectorSpace]]) -> None:
        if type(subspaces) is not list:
            subspaces = [subspaces]
        for subspace in subspaces:
            if subspace.label in self._set:
                if isinstance(self._set[subspace.label], list):
                    self._set[subspace.label].append(subspace)
                else:
                    self._set[subspace.label] = [self._set[subspace.label], subspace]
            else:
                self._set[subspace.label] = subspace

    def __len__(self) -> int:
        """
        Number of subspaces with different labels
        """
        return len(self._set)

    def __getitem__(self, label) -> VectorSpace:
        if label in self._set:
            return self._set[label]
        raise (IndexError(f"Label {label} not in VectorSet"))
