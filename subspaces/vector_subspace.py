import torch
import unittest


class VectorSubspace:
    """
    Class that defines a simple subspace
    """
    def __init__(self, n:int=0, vector_size:int=0):
        """
        n (int): number of vectors in subspace
        vector_size (int): size of vector in subspace
        """
        self.n = n
        self.vector_size = vector_size
        self.A = torch.rand(self.n, self.vector_size)
    
    def __len__(self):
        return self.n

    def __getitem__(self, i:int):
        if i < self.n:
            return self.A[i]
        raise(IndexError("Index i out of bound"))

    def append(self, vector:torch.Tensor):
        if vector.dim() == 1:
            vector.unsqueeze_(0)
        assert(vector.shape[1] == self.vector_size)
        self.n = self.n + vector.shape[0]
        self.A = torch.cat([self.A, vector], dim=0)
    
    def svd(self, full_matrices=False):
        U, S, Vh = torch.linalg.svd(self.A, full_matrices=full_matrices)
        return U, S, Vh


# --- unittests
class TestVectorSubspace(unittest.TestCase):
    def test_init(self):
        _ = VectorSubspace()

    def test_getitem_len(self):
        subspace = VectorSubspace(10, 32)
        vector = subspace[1]
        self.assertEqual(vector.shape[0], 32)
        self.assertEqual(len(subspace), 10)
        with self.assertRaises(IndexError):
            subspace[11]
    
    def test_svd(self):
        subspace = VectorSubspace(10,32)
        U, S, V = subspace.svd()
    
    def test_append(self):
        subspace = VectorSubspace(10, 32)
        vector_1 = torch.rand(32)
        subspace.append(vector_1)
        assert(torch.allclose(subspace[10], vector_1))

        vector_10 = torch.rand(10, 32)
        subspace.append(vector_10)
        assert(torch.allclose(subspace[20], vector_10[9]))



if __name__ == "__main__":
    unittest.main()