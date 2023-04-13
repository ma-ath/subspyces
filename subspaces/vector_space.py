import math
import torch
import unittest


class VectorSpace:
    """
    Class that defines a simple subspace
    """
    def __init__(self, n:int=0, vector_size:int=0, label=None):
        """
        n (int): number of vectors in subspace
        vector_size (int): size of vector in subspace
        """
        self.n = n
        self.vector_size = vector_size
        self.A = torch.rand(self.n, self.vector_size)
        self.label = label
    
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
        vector = vector.type(torch.FloatTensor)
        self.n = self.n + vector.shape[0]
        self.A = torch.cat([self.A, vector], dim=0)
    
    def svd(self, full_matrices=False):
        U, S, Vh = torch.linalg.svd(self.A, full_matrices=full_matrices)
        return U, S, Vh

    def pca(self, min_energy:float=0.8):
        # calculate mean vector
        # mean_vector = self.A.mean(dim=0)

        # standardize data matrix
        # A_pca = torch.sub(self.A, mean_vector)
        # A_pca = torch.div(A_pca, math.sqrt(self.vector_size))

        # Calculate SVD
        U, S, Vh = self.svd(full_matrices=False)

        # Get base vectors for subspace from min_energy
        cumulative_energy = torch.cumsum(S, dim=0) / torch.sum(S)
        for i, energy in enumerate(cumulative_energy):
            n = i+1
            if energy >= min_energy:
                break
            
        # Generate Subspace
        subspace = VectorSpace(vector_size=self.vector_size)
        subspace.append(Vh[:n])

        return subspace

    def __lt__(self, other):
         return self.n < other.n
    
    def __str__(self):
        return f"VectorSpace:{self.n}x{self.vector_size}"

    def __repr__(self):
        return f"VectorSpace:{self.n}x{self.vector_size}"


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