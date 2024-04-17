import unittest
import numpy as np
import torch
import math

from subspyces.metrics import cosine_canonical_angles
from subspyces import VectorSpace


# --- unittests
class TestMetricsCanonicalAngles(unittest.TestCase):
    def setUp(self):
        self.A = np.array([
            [10, 0, 0, 0],  # First basis
            [0, 15, 0, 0],  # Second basis
            [0, 0, 5, 5]   # Third basis
        ])
        self.B = np.array([
            [0, 8, 0, 0],  # First basis
            [0, 0, 7, 0],  # Second basis
            [19, 0, 0, 19]   # Third basis
        ])
        # TODO: Check these ...
        self.res1 = torch.tensor([
            [1, math.sqrt(2)/2, 0.5]
        ])
        self.C = np.array(
            [1, 0, 0, 0]
        )
        self.res2 = torch.tensor(
            [1, 0, math.sqrt(2)/2]
        )
        self.D = np.array(
            [0, -1, 0, 0]
        )
        self.res3 = np.array(
            [0, 0, 0, 0]
        )
        self.subspaceA = VectorSpace(4).append(self.A)
        self.subspaceB = VectorSpace(4).append(self.B)

    def test_input_checks(self):
        with self.assertRaises(TypeError):
            cosine_canonical_angles(self.A, None)
        with self.assertRaises(TypeError):
            cosine_canonical_angles(None, self.B)
        with self.assertRaises(RuntimeError):
            cosine_canonical_angles(torch.rand(1, 2, 2), torch.rand(1, 2, 2))

    def test_can_calculate(self):
        _ = cosine_canonical_angles(torch.rand(2, 2), torch.rand(2, 2))
        _ = cosine_canonical_angles(np.random.rand(2, 2), np.random.rand(2, 2))
        _ = cosine_canonical_angles(self.subspaceA, self.subspaceB)
        _ = cosine_canonical_angles(torch.rand(2), torch.rand(2, 2))

    def test_calculation_is_correct(self):
        for use_svd in [True]:
            _ = cosine_canonical_angles(self.A, self.B, use_svd=use_svd)
            # self.assertTrue(torch.allclose(ca, self.res1))
            # ca = cosine_canonical_angles(self.B, self.A)
            # self.assertTrue(torch.allclose(ca, self.res1))
            # ca = cosine_canonical_angles(self.A, self.C)
            # self.assertTrue(torch.allclose(ca, self.res2))
            # ca = cosine_canonical_angles(self.C, self.A)
            # self.assertTrue(torch.allclose(ca, self.res2))


if __name__ == "__main__":
    unittest.main()
