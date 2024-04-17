import torch
from torch import functional as F
from typing import Union
import numpy as np

from subspyces import VectorSpace


def cosine_similarity(x: Union[torch.Tensor, np.ndarray, VectorSpace],
                      y: Union[torch.Tensor, np.ndarray, VectorSpace]) -> torch.Tensor:
    r"""
    Returns the cosine similarity between basis vectors of subspaces
    :math:`cs = \frac{(\phi_i,\psi_j)}{\|\phi_i\|\|\psi_j\|}`
    """
    if (not isinstance(x, (torch.Tensor, np.ndarray, VectorSpace)) or
            not isinstance(y, (torch.Tensor, np.ndarray, VectorSpace))):
        raise (TypeError("Invalid input type!"))
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, VectorSpace):
        x = x._data
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    elif isinstance(y, VectorSpace):
        y = y._data
    if x.dim() > 2 or y.dim() > 2:
        raise (RuntimeError(("input cannot have more then 2 dimensions, "
                            f"but is has {x.dim()} and {y.dim()} dimensions.")))

    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)

    cosine_similarity = torch.matmul(F.normalize(x),
                                     F.normalize(y).H)
    return cosine_similarity
