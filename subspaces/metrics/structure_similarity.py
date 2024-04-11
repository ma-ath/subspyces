import torch
from torch import linalg
from torch.nn import functional as F

from subspaces import VectorSpace


def structure_similarity(A: VectorSpace, B: VectorSpace) -> float:
    if A.dim != B.dim:
        raise (AssertionError(f"VectorSpaces have different dimension! {A.dim} and {B.dim}"))

    similarity_matrix = F.normalize(A._data) @ F.normalize(B._data).T

    squared_cossines = linalg.svdvals(similarity_matrix)

    return float(torch.mean(squared_cossines))
