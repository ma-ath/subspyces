import torch
from torch import linalg
from sklearn.decomposition import PCA
import warnings
from warnings import WarningMessage

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
        self._etol = 1e-6

    def __str__(self) -> str:
        return "PCATransform"

    def transform(self, vector_space: VectorSpace, *args, **kwargs) -> VectorSpace:
        if self.use_sklearn:
            pca_ = self._pca_transform.fit(vector_space._data).components_.copy()
        else:
            auto_correlation_matrix = vector_space._data.T.conj() @ vector_space._data
            auto_correlation_matrix /= (vector_space.n-1)
            eigenvalues, eigenvectors = linalg.eig(auto_correlation_matrix)
            #  Eigenvalues are supposed to be positive reals.
            # This condtion may fail only if there are numerical errors. 
            # We can ignore the torch warning here about casting complex -> real
            if torch.max(torch.imag(eigenvalues)) > self._etol:
                raise (AssertionError(
                    "Eigenvalues of autocorrelation matrix are supposed to be real," +
                    f"but has imaginary part {torch.max(torch.imag(eigenvalues))}"))

            # TODO: This filter is not working...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eigenvalues = torch.real(eigenvalues)

            sorted_indices = torch.argsort(eigenvalues, descending=True)
            pca_ = eigenvectors[sorted_indices, :][:self.n_components]

            # Check the eigenvectors for NaN. An NaN can show there was a numerical error
            if torch.isnan(pca_).any():
                warnings.warn("A NaN was generated while calculating the eigenvectors." +
                              "This may suggest a numerical error on calculation." +
                              "Casting all NaN to 0.", RuntimeWarning)

                pca_ = torch.nan_to_num(pca_)

        new_vector_space = VectorSpace(dim=vector_space.dim, label=vector_space.label)
        new_vector_space.append(pca_)
        return new_vector_space
