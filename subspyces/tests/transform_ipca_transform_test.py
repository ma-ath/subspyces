import unittest
import torch
import torchvision.transforms as T
# from torchvision.datasets import FakeData
from torchvision.datasets import MNIST

from subspyces.transform import IncrementalPCATransform
from subspyces.transform import PCATransform
from subspyces.generators import IdentityGenerator


# --- unittests
class TestPCATransform(unittest.TestCase):
    def setUp(self):
        # dataset = FakeData(64, [32, 32], 2,
        #                    transform=T.Compose([T.PILToTensor(), torch.flatten]))
        dataset = MNIST("~/datasets", download=True, train=False,
                        transform=T.Compose([T.ToTensor(), torch.flatten]))
        generator = IdentityGenerator()

        self.vector_spaces = generator.generate(dataset, batch_size=32)

    def test_transform(self):
        for use_sklearn in [True]:  # , False]:
            pca_transform = IncrementalPCATransform(n_components=10, batch_size=32,
                                                    use_sklearn=use_sklearn)
            for vector_space in self.vector_spaces.values():
                _ = pca_transform.transform(vector_space)

    def test_pca_ipca_close(self):
        etol = 1e-4
        for use_sklearn in [True]:  # , False]:
            ipca_transform = IncrementalPCATransform(n_components=10, batch_size=None,
                                                     use_sklearn=use_sklearn)
            pca_transform = PCATransform(n_components=10, use_sklearn=use_sklearn)
            for vector_space in self.vector_spaces.values():
                pca = pca_transform.transform(vector_space)
                ipca = ipca_transform.transform(vector_space)
                error = torch.abs(torch.abs(pca._data) - torch.abs(ipca._data)).mean()
                print(error)
                self.assertTrue(error < etol)


if __name__ == "__main__":
    unittest.main()
