import unittest
import torch
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import Subset

from subspyces.methods import SubspaceMethod


class TestMethodsSubspaceMethod(unittest.TestCase):
    def setUp(self):
        self.train_dataset = Subset(MNIST("~/datasets", download=True, train=True,
                                          transform=T.Compose([T.ToTensor(), torch.flatten])),
                                    indices=range(1000))
        self.eval_dataset = Subset(MNIST("~/datasets", download=True, train=False,
                                         transform=T.Compose([T.ToTensor(), torch.flatten])),
                                   indices=range(1000))
        self.model = SubspaceMethod()

    def test_model(self):
        # Run an MNIST classification and check the results
        self.model.train(self.train_dataset, 10)
        classification_ration = self.model.eval(self.eval_dataset)
        print(classification_ration)
        self.assertTrue(classification_ration > 0.85)


if __name__ == "__main__":
    unittest.main()
