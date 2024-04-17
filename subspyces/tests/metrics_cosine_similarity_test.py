import unittest
import numpy as np
import torch
import math

from subspyces.metrics import cosine_similarity
from subspyces import VectorSpace


# --- unittests
class TestMetricsCosineSimilarity(unittest.TestCase):
    def setUp(self):
        self.A = np.array([
            [10, 0, 0, 0],  # First basis
            [0, 15, 0, 0],  # Second basis
            [5, 5, 0, 0]   # Third basis
        ])
        self.B = np.array([
            [0, 0, 8, 0],  # First basis
            [0, 7, 0, 0],  # Second basis
            [0, 19, 19, 0]   # Third basis
        ])
        self.res1 = torch.tensor([
            [0, 0, 0],
            [0, 1, math.sqrt(2)/2],
            [0, math.sqrt(2)/2, 0.5]
        ])
        self.C = np.array(
            [1, 0, 0, 0]
        )
        self.res2 = torch.tensor(
            [1, 0, math.sqrt(2)/2]
        )
        self.subspaceA = VectorSpace(4).append(self.A)
        self.subspaceB = VectorSpace(4).append(self.B)

    def test_input_checks(self):
        with self.assertRaises(TypeError):
            cosine_similarity(self.A, None)
        with self.assertRaises(TypeError):
            cosine_similarity(None, self.B)
        with self.assertRaises(RuntimeError):
            cosine_similarity(torch.rand(1, 2, 2), torch.rand(1, 2, 2))

    def test_can_calculate(self):
        _ = cosine_similarity(torch.rand(2, 2), torch.rand(2, 2))
        _ = cosine_similarity(np.random.rand(2, 2), np.random.rand(2, 2))
        _ = cosine_similarity(self.subspaceA, self.subspaceB)
        _ = cosine_similarity(torch.rand(2), torch.rand(2, 2))

    def test_calculation_is_correct(self):
        cs = cosine_similarity(self.A, self.B)
        self.assertTrue(torch.allclose(cs, self.res1))
        cs = cosine_similarity(self.B, self.A)
        self.assertTrue(torch.allclose(cs, self.res1))
        cs = cosine_similarity(self.A, self.C)
        self.assertTrue(torch.allclose(cs, self.res2))
        cs = cosine_similarity(self.C, self.A)
        self.assertTrue(torch.allclose(cs, self.res2))


if __name__ == "__main__":
    unittest.main()
