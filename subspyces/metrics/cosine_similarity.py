import torch
from torch.nn import functional as F
from typing import Union
import numpy as np

from subspyces.core import VectorSpace


def cosine_similarity(x: Union[torch.Tensor, np.ndarray, VectorSpace],
                      y: Union[torch.Tensor, np.ndarray, VectorSpace]) -> torch.Tensor:
    r"""
    Returns a matrix of cosine similarities between the basis vectors of subspaces
    :math:`cs = \frac{(\phi_i,\psi_j)}{\|\phi_i\|\|\psi_j\|}`
    """
    # NOTE: These many checks can slow down computation. Maybe there is a better way to
    # implement this generically (maybe just ignore the checks?).
    if (not isinstance(x, (torch.Tensor, np.ndarray, VectorSpace)) or
            not isinstance(y, (torch.Tensor, np.ndarray, VectorSpace))):
        raise (TypeError(f"Invalid input type {type(x)} or {type(y)}"))
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
    if x.dim() == 1:
        x.unsqueeze_(0)
    if y.dim() == 1:
        y.unsqueeze_(0)

    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)

    cosine_similarity = torch.matmul(F.normalize(x),
                                     F.normalize(y).H)
    cosine_similarity.squeeze_()
    return cosine_similarity
