import unittest
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Subset

from subspyces.transform import PCATransform
from subspyces.generators import IdentityGenerator
from subspyces.core import VectorSpace


# --- unittests
class TestPCATransform(unittest.TestCase):
    def setUp(self):
        dataset = Subset(MNIST("~/datasets", download=True, train=False,
                               transform=T.Compose([T.ToTensor(), torch.flatten])),
                         indices=range(1000))
        generator = IdentityGenerator()

        self.vector_spaces = generator.generate(dataset, batch_size=32)

    def test_transform_n_components(self):
        pca_transform = PCATransform(n_components=10)
        for vector_space in self.vector_spaces.values():
            _ = pca_transform.transform(vector_space)

    def test_transform_min_energy(self):
        pca_transform = PCATransform(min_energy=0.95)
        for vector_space in self.vector_spaces.values():
            _ = pca_transform.transform(vector_space)

    def test_transform_1_base_subspace(self):
        pca_transform = PCATransform(min_energy=0.95, use_sklearn=True)

        vector_space = VectorSpace(dim=32)
        vector_space.append(torch.randn(32))

        pca_transform.transform(vector_space)


if __name__ == "__main__":
    unittest.main()
