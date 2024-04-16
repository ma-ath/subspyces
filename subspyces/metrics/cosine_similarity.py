import torch
from torch import linalg
from typing import Union
import numpy as np

from subspyces import VectorSpace


def cosine_similarity(x: Union[torch.Tensor, np.ndarray, VectorSpace],
                      y: Union[torch.Tensor, np.ndarray, VectorSpace]) -> torch.Tensor:
    raise (NotImplementedError)
    """
    Returns S = \sum_{i=0}^{r-1} \frac{(x,\phi_i)^2}{\|x\|\|\phi_i\|}
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
