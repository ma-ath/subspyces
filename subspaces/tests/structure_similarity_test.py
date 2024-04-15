import unittest
import torch
import torchvision.transforms as T
# from torchvision.datasets import FakeData
from torchvision.datasets import MNIST

from subspaces.metrics import structure_similarity
from subspaces.generators import IdentityGenerator
from subspaces.transform import PCATransform


# --- unittests
class TestPCATransform(unittest.TestCase):
    def setUp(self):
        # dataset = FakeData(64, [32, 32], 2,
        #                    transform=T.Compose([T.ToTensor(), torch.flatten]))
        dataset = MNIST("~/datasets", download=True, train=False,
                        transform=T.Compose([T.ToTensor(), torch.flatten]))
        generator = IdentityGenerator()
        vector_spaces = generator.generate(dataset, batch_size=32)
        pca_transform = PCATransform(n_components=10)
        self.data = dict()
        for vector_space in vector_spaces.values():
            self.data[vector_space.label] = pca_transform.transform(vector_space)

    def is_psd(self, mat):
        return bool(torch.allclose(mat, mat.H) and (torch.linalg.eigvals(mat).real >= 0).all())

    def test_structure_similarity(self):
        gram_matrix = torch.zeros(10, 10)
        for i in range(0, 10):
            for j in range(0, 10):
                gram_matrix[i][j] = structure_similarity(self.data[i], self.data[j])
        self.assertTrue(self.is_psd(gram_matrix))


if __name__ == "__main__":
    unittest.main()
