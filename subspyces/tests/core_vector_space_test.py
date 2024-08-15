import torch
import numpy as np
import unittest

from subspyces.core import VectorSpace


# --- unittests
class TestVectorSpace(unittest.TestCase):
    def setUp(self):
        self.dim = 32

    def test_init(self):
        _ = VectorSpace(dim=10)

    def test_append(self):
        subspace = VectorSpace(dim=self.dim)
        subspace.append(torch.rand(self.dim))
        self.assertEqual(len(subspace), 1)

        subspace.append(np.random.rand(self.dim))
        self.assertEqual(len(subspace), 2)

        subspace.append(np.random.rand(8, self.dim))
        self.assertEqual(len(subspace), 10)

        with self.assertRaises(TypeError):
            subspace.append(None)

        with self.assertRaises(AssertionError):
            subspace.append(np.random.rand(5, 5, self.dim))

        with self.assertRaises(AssertionError):
            subspace.append(np.random.rand(self.dim+1))

    def test_getitem(self):
        subspace = VectorSpace(dim=self.dim)

        self.assertEqual(len(subspace), 0)

        with self.assertRaises(IndexError):
            subspace[1]
        with self.assertRaises(IndexError):
            subspace[-1]

        vector_1 = torch.rand(self.dim)
        subspace.append(vector_1)
        self.assertTrue(torch.allclose(subspace[0], vector_1))

        vector_10 = torch.rand(10, self.dim)
        subspace.append(vector_10)
        self.assertTrue(torch.allclose(subspace[1:11], vector_10))

    def test_lt(self):
        vspace1 = VectorSpace(dim=self.dim)
        vspace1.append(torch.rand(self.dim))

        vspace2 = VectorSpace(dim=self.dim)
        vspace2.append(torch.rand(self.dim))
        vspace2.append(torch.rand(self.dim))

        vspace3 = VectorSpace(dim=self.dim)
        vspace3.append(torch.rand(self.dim))
        vspace3.append(torch.rand(self.dim))
        vspace3.append(torch.rand(self.dim))

        foo_list = [vspace2, vspace1, vspace3]

        self.assertEqual(sorted(foo_list), [vspace1, vspace2, vspace3])


if __name__ == "__main__":
    unittest.main()
