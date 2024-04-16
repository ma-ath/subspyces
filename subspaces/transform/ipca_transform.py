import torch
from torch import linalg
from sklearn.decomposition import IncrementalPCA
import warnings

from .abstract_transform import AbstractTransform
from subspaces import VectorSpace


class IncrementalPCATransform(AbstractTransform):
    """
    Applies the sklearn.decomposition.PCA onto a VectorSpace
    """
    def __init__(self, n_components: int, use_sklearn: bool = False, **kwargs):
        self.use_sklearn = use_sklearn
        self.n_components = n_components
        if self.use_sklearn:
            self._ipca_transform = IncrementalPCA(n_components=self.n_components,
                                                  copy=False,
                                                  **kwargs)
        self._etol = 1e-6

    def __str__(self) -> str:
        return "IncrementalPCATransform"

    def transform(self, vector_space: VectorSpace, *args, **kwargs) -> VectorSpace:
        if self.use_sklearn:
            warnings.warn("sklearn calculates the PCA sligthly different then general subspaces. " +
                          "Make sure you know what you are doing", UserWarning)
            ipca_ = self._ipca_transform.fit(vector_space._data).components_.copy().T
        else:
            auto_correlation_matrix = torch.matmul(vector_space._data.H, vector_space._data)
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
                eigenvectors = torch.real(eigenvectors)

            sorted_indices = torch.argsort(eigenvalues, descending=True)
            ipca_ = eigenvectors[:, sorted_indices][:, :self.n_components]

            # Check the eigenvectors for NaN. An NaN can show there was a numerical error
            if torch.isnan(ipca_).any():
                warnings.warn("A NaN was generated while calculating the eigenvectors." +
                              "This may suggest a numerical error on calculation." +
                              "Casting all NaN to 0.", RuntimeWarning)

                ipca_ = torch.nan_to_num(ipca_)

        new_vector_space = VectorSpace(dim=vector_space.dim, label=vector_space.label)
        new_vector_space.append(ipca_.T)
        return new_vector_space
