import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union
import numpy as np

from subspyces import VectorSpace
from subspyces.generators import IdentityGenerator
from subspyces.transform import PCATransform
from subspyces.metrics import cosine_similarity
from . import AbstractMethod


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

        correct_class = []
        classification_ratio = 0

        for n_batch, (batch_data, batch_label) in enumerate(loader):
            predicted_labels = self.classify(batch_data)

            for l1, l2 in zip(predicted_labels, batch_label):
                correct_class.append(l1 == l2)

            classification_ratio += correct_class.count(True) / len(correct_class)
        classification_ratio / n_batch

        return correct_class, classification_ratio

    def classify(self, vectors: Union[torch.Tensor, np.ndarray]):
        """
        Classifies a tensor in one of the subspaces using the mean squared of canonical angles
        """
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        # Classify all vectors by cossine similarity
        max_likelihood = []
        cs = []

        for subspace in self._model_parameters:
            cs = cosine_similarity(vectors, subspace)
            mscca = torch.mean(torch.square(cs), dim=1)
            for i in mscca:
                if foo[i] > cs[i]: cs[i] = foo[i]; max_likelihood[i] = subspace.label
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
