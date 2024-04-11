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
        pca_data = self._pca_transform.fit_transform(vector_space._data)
        pca_vspace = VectorSpace(dim=vector_space.dim, label=vector_space.label)
        pca_vspace.append(pca_data)
        return pca_vspace
