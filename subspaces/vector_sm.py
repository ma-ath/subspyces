import torch
import unittest

from vector_subspace import VectorSubspace
from vector_set import VectorSet


class VectorSM:
    """
    Class that implements a simple subspace method for vector subspaces
    """
    def __init__(self, vector_size:int=1) -> None:
        self.vector_size = vector_size
        pass

    def train(self):
        pass

    def eval(self):
        pass

# --- unittests
class TestVectorSM(unittest.TestCase):
    def test_init(self):
        pass
        

if __name__ == "__main__":
    unittest.main()