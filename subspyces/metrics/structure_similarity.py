import torch
from torch import linalg
from torch.nn import functional as F
from typing import Union
import numpy as np

from subspyces import VectorSpace


def structure_similarity(A: Union[torch.Tensor, np.ndarray, VectorSpace],
                         B: Union[torch.Tensor, np.ndarray, VectorSpace]) -> float:
    r"""
    Structure similarity between two vector spaces.
    This returns the mean of squared cosines of the canonical angles between
    subspaces :math:`\mathcal{A}` and :math:`\mathcal{B}`. Canonical angles
    are calculated using the SVD of the similarity matrix.
    """
    if A.dim != B.dim:
        raise (AssertionError(f"VectorSpaces have different dimension! {A.dim} and {B.dim}"))

    similarity_matrix = torch.matmul(F.normalize(A._data),
                                     F.normalize(B._data).H)

    canonical_angles = linalg.svdvals(similarity_matrix)

    return float(torch.mean(torch.square(canonical_angles)))
