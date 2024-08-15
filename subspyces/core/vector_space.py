import torch
import numpy as np
from typing import Union, Any


class VectorSpace:
    """
    Class that defines a simple vector space
    """
    def __init__(self, *, dim: int, label: Any = None) -> None:
        """
        dim (int): dimension of vectors in space
        label (Any): A label for the vector space
        """
        self.dim = dim
        self.n = 0
        self.label = label
        self.dtype = torch.FloatTensor
        self._data = torch.rand(self.n, self.dim)

    def __len__(self) -> int:
        """
        Returns the number of bases in vector space
        """
        return self.n

    def __getitem__(self, i: Union[int, slice]) -> torch.Tensor:
        """
        returns the i'th base of vector space
        """
        if type(i) is slice:
            if i.stop > self.n or i.start < 0:
                raise (IndexError("Index i out of bound"))
        elif i >= self.n and i < 0:
            raise (IndexError("Index i out of bound"))
        return self._data[i]

    def to(self, device: Union[torch.device, str]) -> "VectorSpace":
        """
        Sends vector space to specific device
        """
        self._data.to(device=device)
        return self

    def append(self, vector: Union[torch.Tensor, np.ndarray]) -> "VectorSpace":
        """
        Appends vectors to the basis vectors of vector space
        """
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
        return self

    def __lt__(self, other) -> int:
        return self.n < other.n

    def __str__(self) -> str:
        return f"VectorSpace:{self.n}x{self.dim}"

    def __repr__(self) -> str:
        return self.__str__()
