import torch
from torch import linalg
from sklearn.decomposition import PCA
import warnings

from .abstract_transform import AbstractTransform
from subspyces.core import VectorSpace


class PCATransform(AbstractTransform):
    """
    Applies the sklearn.decomposition.PCA onto a VectorSpace
    """
    def __init__(self, *,
                 n_components: int = None,
                 min_energy: float = None,
                 use_sklearn: bool = False, **kwargs):
        self.use_sklearn = use_sklearn
        self.n_components = n_components
        self.min_energy = min_energy
        if self.n_components is None and self.min_energy is None:
            raise ValueError("You must specify either n_components or min_energy")
        if self.n_components is not None and self.min_energy is not None:
            raise ValueError("You must specify either n_components or min_energy, but not both")
        if self.n_components is not None:
            assert self.n_components > 0, "n_components must be greater than 0"
        if self.min_energy is not None:
            assert 0 < self.min_energy <= 1, "min_energy must be in interval (0, 1]"
        if self.use_sklearn:
            self._pca_transform = PCA(n_components=self.n_components,
                                      copy=False,
                                      **kwargs)
        self._etol = 1e-3

    def __str__(self) -> str:
        return "PCATransform"

    def transform(self, vector_space: VectorSpace, *args, **kwargs) -> VectorSpace:
        if vector_space.n < 2:
            warnings.warn("The input VectorSpace only have one basis vector. Calculated 'PCA' is "
                          "simply this basis vector normalized.", UserWarning)
            pca_ = vector_space._data / torch.norm(vector_space._data)
            pca_ = pca_.T
        elif self.use_sklearn:
            warnings.warn("sklearn calculates the PCA sligthly different then general subspyces. "
                          "Make sure you know what you are doing", UserWarning)
            pca_ = self._pca_transform.fit(vector_space._data).components_.copy().T
        else:
            auto_correlation_matrix = torch.matmul(vector_space._data.H, vector_space._data)
            auto_correlation_matrix /= (vector_space.n-1)
            eigenvalues, eigenvectors = linalg.eig(auto_correlation_matrix)
            #  Eigenvalues are supposed to be positive reals.
            # This condtion may fail only if there are numerical errors.
            # We can ignore the torch warning here about casting complex -> real
            if torch.max(torch.imag(eigenvalues)) > self._etol:
                raise (AssertionError(
                    "Eigenvalues of autocorrelation matrix are supposed to be real, "
                    f"but has imaginary part {torch.max(torch.imag(eigenvalues))}"))

            # BUG: This filter is not working...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eigenvalues = torch.real(eigenvalues)
                eigenvectors = torch.real(eigenvectors)

            squared_eigenvalues = torch.square(eigenvalues)
            sorted_indices = torch.argsort(squared_eigenvalues, descending=True)
            squared_eigenvalues.sort(descending=True)
            if self.n_components is not None:
                pca_ = eigenvectors[:, sorted_indices][:, :self.n_components]
            elif self.min_energy is not None:
                cumulative_energy = torch.cumsum(squared_eigenvalues,
                                                 dim=0) / torch.sum(squared_eigenvalues)
                n_components = torch.sum(cumulative_energy <= self.min_energy) + 1
                pca_ = eigenvectors[:, sorted_indices][:, :n_components]
            else:
                raise RuntimeError("Unexpected error. This should not happen.")

            # Check the eigenvectors for NaN. An NaN can show there was a numerical error
            if torch.isnan(pca_).any():
                warnings.warn("A NaN was generated while calculating the eigenvectors. "
                              "This may suggest a numerical error on calculation. "
                              "Casting all NaN to 0.", RuntimeWarning)

                pca_ = torch.nan_to_num(pca_)

        new_vector_space = VectorSpace(dim=vector_space.dim, label=vector_space.label)
        new_vector_space.append(pca_.T)
        return new_vector_space
