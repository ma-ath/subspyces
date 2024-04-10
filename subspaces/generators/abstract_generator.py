from torch.utils.data import Dataset
from typing import Union

from subspaces import VectorSpace, VectorSet


class AbstractGenerator:
    """
    Abstract base class for subspace generators

    A subspace generator receives a torch dataset and outputs a VectorSpace (unique label),
    or VectorSet (multiple labels).
    """
    def __init__(self):
        raise (NotImplementedError)

    def __str__(self) -> str:
        raise (NotImplementedError)

    def generate(self, dataset: Dataset, *args, **kwargs) -> Union[VectorSpace, VectorSet]:
        raise (NotImplementedError)
