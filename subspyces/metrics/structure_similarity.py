import torch
from typing import Union
import numpy as np

from subspyces.core import VectorSpace
from .cosine_canonical_angles import cosine_canonical_angles


def structure_similarity(A: Union[torch.Tensor, np.ndarray, VectorSpace],
                         B: Union[torch.Tensor, np.ndarray, VectorSpace]) -> torch.Tensor:
    r"""
    Structure similarity between two vector spaces.
    This returns the mean of squared cosines of the canonical angles between
    subspaces :math:`\mathcal{A}` and :math:`\mathcal{B}`. Canonical angles
    are calculated using the SVD of the similarity matrix.
    """
    ca = cosine_canonical_angles(A, B)
    return torch.mean(torch.square(ca))
