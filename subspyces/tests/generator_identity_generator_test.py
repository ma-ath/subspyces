import unittest
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Subset

from subspyces.generators import IdentityGenerator


# --- unittests
class TestIdentityGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = IdentityGenerator()
        self.dataset = Subset(MNIST("~/datasets", download=True, train=False,
                                    transform=T.Compose([T.PILToTensor(), torch.flatten])),
                              indices=range(1000))

    def test_generate(self):
        _ = self.generator.generate(self.dataset, batch_size=32)


if __name__ == "__main__":
    unittest.main()
