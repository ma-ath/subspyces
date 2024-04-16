from torch.utils.data import Dataset
from typing import Dict, Any

from subspyces import VectorSpace


class AbstractGenerator:
    """
    Abstract base class for subspace generators

    A subspace generator receives a torch dataset and outputs a dictionary of VectorSpace.
    """
    def __init__(self):
        raise (NotImplementedError)

    def __str__(self) -> str:
        raise (NotImplementedError)

    def generate(self, dataset: Dataset, *args, **kwargs) -> Dict[Any, VectorSpace]:
        raise (NotImplementedError)
