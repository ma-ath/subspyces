import torch
from torch import linalg
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
    if vector.dim() == 1:
        vector.unsqueeze_(0)
    assert(vector.shape[1] == subspace.vector_size)
    vector = vector.type(torch.FloatTensor)

    S = torch.sum(
            torch.div(
                torch.mm(vector, subspace.A.t())**2,
                torch.matmul(
                    torch.sqrt(
                        torch.diag(
                            torch.mm(vector, vector.t()
                            )
                        )
                    ).unsqueeze(0).t(),
                    torch.sqrt(
                        torch.diag(
                            torch.mm(subspace.A, subspace.A.t())
                        )
                    ).unsqueeze(0)
                )
            ), dim=1
        )
    print(S)
    return S
    raise (NotImplementedError)
