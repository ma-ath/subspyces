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

    def test_structure_similarity(self):
        _ = structure_similarity(self.data[0], self.data[1])


if __name__ == "__main__":
    unittest.main()
