import torch
from torch import linalg
from typing import Union
import numpy as np

from subspyces import VectorSpace
from .cosine_similarity import cosine_similarity


def canonical_angles(x: Union[torch.Tensor, np.ndarray, VectorSpace],
                     y: Union[torch.Tensor, np.ndarray, VectorSpace],
                     use_svd=True) -> torch.Tensor:
    r"""
    Returns the cosine similarity between basis vectors of subspaces
    :math:`cs = \frac{(\phi_i,\psi_j)}{\|\phi_i\|\|\psi_j\|}`
    """
    if use_svd:
        cs = cosine_similarity(x, y)
        canonical_angles = linalg.svdvals(cs)
    else:
        raise (NotImplementedError)
    return canonical_angles
