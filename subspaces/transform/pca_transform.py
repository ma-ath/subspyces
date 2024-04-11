from .abstract_transform import AbstractTransform
from subspaces import VectorSpace

from sklearn.decomposition import PCA


class PCATransform(AbstractTransform):
    """
    Applies the sklearn.decomposition.PCA onto a VectorSpace
    """
    def __init__(self, n_components: int, whiten: bool = False,
                 svd_solver: str = 'arpack', **kwargs):
        self._pca_transform = PCA(n_components=n_components,
                                  copy=False,
                                  whiten=whiten,
                                  svd_solver=svd_solver,
                                  **kwargs)

    def __str__(self) -> str:
        return "PCATransform"

    def transform(self, vector_space: VectorSpace, *args, **kwargs) -> VectorSpace:
        pca_ = self._pca_transform.fit(vector_space._data)
        new_vector_space = VectorSpace(dim=vector_space.dim, label=vector_space.label)
        new_vector_space.append(pca_.components_.copy())
        return new_vector_space
