import torch
from torch import linalg
from typing import Union
import numpy as np

from subspyces.core import VectorSpace
from .cosine_similarity import cosine_similarity


def cosine_canonical_angles(x: Union[torch.Tensor, np.ndarray, VectorSpace],
                            y: Union[torch.Tensor, np.ndarray, VectorSpace],
                            use_svd=True) -> torch.Tensor:
    r"""
    Returns the cosine of canonical angles between basis vectors of subspaces
    :math:`cs = \frac{(\phi_i,\psi_j)}{\|\phi_i\|\|\psi_j\|}`

    These are not squared cosines! Square these values by using torch.square()
    """
    if use_svd:
        # NOTE: SVD can only be applied if (x, y) are ortonormal basis!
        # We do not check for ortonormality, so make sure you know what you are doing.
        # Normalization is done automatically here, but not ortogonalization.
        cs = cosine_similarity(x, y)
        if cs.dim() <= 1:
            return cs
        cca = linalg.svdvals(cs)
        cca.sort(descending=True)
        return cca
    else:
        raise (NotImplementedError)
