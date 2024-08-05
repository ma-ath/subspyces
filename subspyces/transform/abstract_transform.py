from subspyces.core import VectorSpace


class AbstractTransform:
    """
    Abstract base class for subspace transform

    A subspace transform receives a 'VectorSpace' and outputs a VectorSpace
    """
    def __init__(self):
        raise (NotImplementedError)

    def __str__(self) -> str:
        raise (NotImplementedError)

    def transform(self, vector_space: VectorSpace, *args, **kwargs) -> VectorSpace:
        raise (NotImplementedError)
