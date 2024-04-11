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
        #                    transform=T.Compose([T.PILToTensor(), torch.flatten]))
        self.data = dict()
        dataset = MNIST("~/datasets", download=True, train=False,
                        transform=T.Compose([T.PILToTensor(), torch.flatten]))
        generator = IdentityGenerator()
        vector_spaces = generator.generate(dataset, batch_size=32)
        pca_transform = PCATransform(n_components=10)
        for vector_space in vector_spaces.values():
            self.data[vector_space.label] = pca_transform.transform(vector_space)

    def is_psd(self, mat):
        print(torch.allclose(mat, mat.T))
        print((torch.linalg.eigvals(mat)))
        return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real >= 0).all())

    def test_structure_similarity(self):
        gram_matrix = torch.zeros(10, 10)
        print(self.data[0])
        # for i in range(1, 10):
        #     for j in range(1, 10):
        #         gram_matrix[i][j] = structure_similarity(self.data[i], self.data[j])

        # print(self.is_psd(gram_matrix))


if __name__ == "__main__":
    unittest.main()
