import unittest

from Subspaces import 

# --- unittests
class TestVectorSpace(unittest.TestCase):
    def test_init(self):
        _ = VectorSpace()

    def test_getitem_len(self):
        subspace = VectorSpace(10, 32)
        vector = subspace[1]
        self.assertEqual(vector.shape[0], 32)
        self.assertEqual(len(subspace), 10)
        with self.assertRaises(IndexError):
            subspace[11]
    
    def test_svd(self):
        subspace = VectorSpace(10,32)
        U, S, V = subspace.svd()
    
    def test_append(self):
        subspace = VectorSpace(10, 32)
        vector_1 = torch.rand(32)
        subspace.append(vector_1)
        assert(torch.allclose(subspace[10], vector_1))

        vector_10 = torch.rand(10, 32)
        subspace.append(vector_10)
        assert(torch.allclose(subspace[20], vector_10[9]))
    
    def test_pca(self):
        subspace = VectorSpace(10, 32)
        pca_base = subspace.pca()
        assert(len(pca_base) <= len(subspace))
        pca_base = subspace.pca(min_energy=1)
        assert(len(pca_base) == len(subspace))
    
    def test_lt(self):
        vspace1 = VectorSpace(10)
        vspace2 = VectorSpace(9)
        vspace3 = VectorSpace(0)
        foo_list = [vspace2, vspace1, vspace3]
        self.assertEqual(sorted(foo_list), [vspace3, vspace2, vspace1])


if __name__ == "__main__":
    unittest.main()