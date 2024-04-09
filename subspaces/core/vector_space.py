import torch
import numpy as np
from typing import Union


class VectorSpace:
    """
    Class that defines a simple subspace
    """
    def __init__(self, dim: int = 0, label=None):
        """
        n (int): number of vectors in subspace
        vector_size (int): size of vector in subspace
        """
        self.dim = dim
        self.n = 0
        self.label = label
        self.dtype = torch.FloatTensor
        self._data = torch.rand(self.n, self.dim)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        if i < self.n and i >= 0:
            return self._data[i]
        raise (IndexError("Index i out of bound"))

    def to(self, device: Union[torch.device, str]):
        self._data.to(device=device)

    def append(self, vector: Union[torch.Tensor, np.ndarray]):
        if type(vector) is np.ndarray:
            vector = torch.from_numpy(vector)
        if not isinstance(vector, torch.Tensor):
            raise (TypeError("Datatype is not supported"))
        if vector.ndim > 2:
            raise (AssertionError("Cannot input tensor of ndim > 2"))
        if vector.dim() == 1:
            vector.unsqueeze_(0)
        assert (vector.shape[1] == self.dim)
        vector = vector.type(self.dtype)
        self.n = self.n + vector.shape[0]
        self._data = torch.cat([self._data, vector], dim=0)

    def __lt__(self, other):
        return self.n < other.n

    def __str__(self):
        return f"VectorSpace:{self.n}x{self.dim}"

    def __repr__(self):
        return f"VectorSpace:{self.n}x{self.dim}"
