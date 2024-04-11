import unittest
import torch
import torchvision.transforms as T
# from torchvision.datasets import FakeData
from torchvision.datasets import MNIST

from subspaces.transform import PCATransform
from subspaces.generators import IdentityGenerator


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
        pca_transform = PCATransform(n_components=10)
        for vector_space in self.vector_spaces.values():
            _ = pca_transform.transform(vector_space)


if __name__ == "__main__":
    unittest.main()
