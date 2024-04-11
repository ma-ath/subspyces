import unittest
import torch
import torchvision.transforms as T
from torchvision.datasets import FakeData
# from torchvision.datasets import MNIST

from subspaces.generators import IdentityGenerator


# --- unittests
class TestIdentityGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = IdentityGenerator()
        # self.dataset = MNIST("~/datasets", download=True, train=False,
        #                      transform=T.Compose([T.PILToTensor(), torch.flatten]))
        self.dataset = FakeData(64, [32, 32], 2,
                                transform=T.Compose([T.PILToTensor(), torch.flatten]))

    def test_generate(self):
        _ = self.generator.generate(self.dataset, batch_size=32)


if __name__ == "__main__":
    unittest.main()