import torch
from torch import linalg
from sklearn.decomposition import PCA

from .abstract_transform import AbstractTransform
from subspaces import VectorSpace


class PCATransform(AbstractTransform):
    """
    Applies the sklearn.decomposition.PCA onto a VectorSpace
    """
    def __init__(self, n_components: int, use_sklearn: bool = False, **kwargs):
        self.use_sklearn = use_sklearn
        self.n_components = n_components
        if self.use_sklearn:
            self._pca_transform = PCA(n_components=self.n_components,
                                      copy=False,
                                      **kwargs)

    def __str__(self) -> str:
        return "PCATransform"

    def transform(self, vector_space: VectorSpace, *args, **kwargs) -> VectorSpace:
        if self.use_sklearn:
            pca_ = self._pca_transform.fit(vector_space._data).components_.copy()
        else:
            auto_correlation_matrix = vector_space._data.T.conj() @ vector_space._data
            auto_correlation_matrix /= (vector_space.n-1)
            eigenvalues, eigenvectors = linalg.eigh(auto_correlation_matrix)
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            pca_ = eigenvectors[sorted_indices, :][1:self.n_components+1]

        new_vector_space = VectorSpace(dim=vector_space.dim, label=vector_space.label)
        new_vector_space.append(pca_)
        return new_vector_space
