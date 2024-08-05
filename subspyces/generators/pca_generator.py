from typing import Dict, Any
from torch.utils.data import Dataset

from . import AbstractGenerator
from subspyces.core import VectorSpace


class IPCAGenerator(AbstractGenerator):
    """
    Abstract base class for subspace generators

    A subspace generator receives a torch dataset and outputs a list of VectorSpace.

    If you need this in your research, consider implementing it here!
    """
    def __init__(self):
        raise (NotImplementedError)

    def __str__(self) -> str:
        raise (NotImplementedError)

    def generate(self, dataset: Dataset, *args, **kwargs) -> Dict[Any, VectorSpace]:
        raise (NotImplementedError)
