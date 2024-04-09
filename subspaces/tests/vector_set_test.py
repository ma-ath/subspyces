import unittest

from subspaces import VectorSpace, VectorSet


# --- unittests
class TestSubspaces(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.test_set = VectorSet([VectorSpace(dim=self.dim, label="A"),
                                   VectorSpace(dim=self.dim, label="B"),
                                   VectorSpace(dim=self.dim, label="C"),
                                   VectorSpace(dim=self.dim, label="C")])

    def test_append(self):
        self.test_set.append([VectorSpace(dim=self.dim, label="D"),
                              VectorSpace(dim=self.dim, label="B"),
                              VectorSpace(dim=self.dim, label="D"),
                              VectorSpace(dim=self.dim, label="C")])

    def test_getitem(self):
        self.assertEqual(type(self.test_set["A"]), VectorSpace)
        self.assertEqual(type(self.test_set["C"]), list)
        self.assertEqual(len(self.test_set["C"]), 2)


if __name__ == "__main__":
    unittest.main()
