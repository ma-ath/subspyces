from typing import Dict, Any
from torch.utils.data import Dataset

from .abstract_generator import AbstractGenerator
from subspyces import VectorSpace


class PcaGenerator(AbstractGenerator):
    """
    Abstract base class for subspace generators

    A subspace generator receives a torch dataset and outputs a list of VectorSpace.
    """
    def __init__(self):
        raise (NotImplementedError)

    def __str__(self) -> str:
        raise (NotImplementedError)

    def generate(self, dataset: Dataset, *args, **kwargs) -> Dict[Any, VectorSpace]:
        raise (NotImplementedError)
