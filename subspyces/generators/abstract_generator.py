from torch.utils.data import Dataset
from typing import Dict, Any

from subspyces.core import VectorSpace


class AbstractGenerator:
    """
    Abstract base class for vector space generators

    A generator receives a torch Dataset and outputs a dictionary of VectorSpace.
    It is the class resposible for interfacing torch with VectorSpace.
    """
    def __init__(self):
        raise (NotImplementedError)

    def __str__(self) -> str:
        raise (NotImplementedError)

    def generate(self, dataset: Dataset, *args, **kwargs) -> Dict[Any, VectorSpace]:
        raise (NotImplementedError)
