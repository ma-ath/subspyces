import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, Any
import numpy as np

from subspyces.generators import IdentityGenerator
from subspyces.transform import PCATransform
from subspyces.metrics import cosine_similarity
from .abstract_method import AbstractMethod


class SubspaceMethod(AbstractMethod):
    """
    Class that implements a simple subspace method for vector subspaces
    """
    def __init__(self) -> None:
        self._model_parameters = None

    def train(self, train_data: Dataset, n_components: int, batch_size: int = 32) -> None:
        """
        Loads a full Dataset to memory into different subspaces.
        Then, applies PCA to each subspace.
        Finally, saves the PCA data to self.__data
        """
        generator = IdentityGenerator()
        pca_transform = PCATransform(n_components=n_components)
        dataset = generator.generate(train_data, batch_size=batch_size)
        self._model_parameters = dict()
        for space in dataset.values():
            pca_subspace = pca_transform.transform(space)
            self._model_parameters[pca_subspace.label] = pca_subspace

    def eval(self, eval_data: Dataset, batch_size: int = 32):
        """
        Simple Method that implements a common evaluation procedure for classification problems
        Returns a list of correct classifications, and the success ratio
        """
        loader = DataLoader(eval_data, batch_size=batch_size)

        classification_ratio = 0

        for n_batch, (batch_data, batch_label) in enumerate(loader):
            correct_class = []
            predicted_labels = self.classify(batch_data)

            for l1, l2 in zip(predicted_labels, batch_label):
                correct_class.append(l1 == l2)

            classification_ratio += correct_class.count(True) / len(batch_data)
        classification_ratio /= (n_batch+1)

        return classification_ratio

    def classify(self, vectors: Union[torch.Tensor, np.ndarray]) -> Any:
        """
        Classifies a tensor in one of the subspaces using the mean squared of canonical angles
        """
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        # Classify all vectors by cossine similarity
        max_likelihood = torch.empty((0, len(vectors)))

        # NOTE: It is faster if you calculate the cosine_similarity of all
        # subspaces in one go. This is not the best way, but it's here
        # iteratively for simplicity.
        for subspace in self._model_parameters.values():
            cs = cosine_similarity(vectors, subspace)
            mscs = torch.mean(torch.square(cs), dim=1).unsqueeze(0)
            max_likelihood = torch.cat([max_likelihood, mscs], 0)
        # See which subspace index the vector is closer to
        max_likelihood = torch.argmax(max_likelihood, dim=0)
        # Transform subspace index to labels
        max_likelihood = [list(self._model_parameters)[i] for i in max_likelihood]
        return max_likelihood

    # def cossine_similarity(self, vector: torch.Tensor, subspace: VectorSpace):
    #     r"""
    #     Returns S = \sum_{i=0}^{r-1} \frac{(x,\phi_i)^2}{\|x\|\|\phi_i\|}
    #     """
    #     if vector.dim() == 1:
    #         vector.unsqueeze_(0)
    #     assert (vector.shape[1] == subspace.vector_size)
    #     vector = vector.type(torch.FloatTensor)

    #     S = torch.sum(
    #             torch.div(
    #                 torch.mm(vector, subspace.A.t())**2,
    #                 torch.matmul(
    #                     torch.sqrt(
    #                         torch.diag(
    #                             torch.mm(vector, vector.t())
    #                         )
    #                     ).unsqueeze(0).t(),
    #                     torch.sqrt(
    #                         torch.diag(
    #                             torch.mm(subspace.A, subspace.A.t())
    #                         )
    #                     ).unsqueeze(0)
    #                 )
    #             ), dim=1
    #         )
    #     return S
