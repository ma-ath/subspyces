import torch
from torch import linalg
from torch.nn import functional as F
from typing import Union
import numpy as np

from subspyces import VectorSpace
from .canonical_angles import canonical_angles


def structure_similarity(A: Union[torch.Tensor, np.ndarray, VectorSpace],
                         B: Union[torch.Tensor, np.ndarray, VectorSpace]) -> float:
    r"""
    Structure similarity between two vector spaces.
    This returns the mean of squared cosines of the canonical angles between
    subspaces :math:`\mathcal{A}` and :math:`\mathcal{B}`. Canonical angles
    are calculated using the SVD of the similarity matrix.
    """
    ca = canonical_angles(A, B)
    return float(torch.mean(torch.square(ca)))
