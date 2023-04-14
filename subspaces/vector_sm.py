import torch
import unittest

from vector_space import VectorSpace
from vector_set import VectorSet


class VectorSM:
    """
    Class that implements a simple subspace method for vector subspaces
    """
    def __init__(self, vector_size:int=1) -> None:
        self.vector_size = vector_size

    def train(self, data:torch.Tensor, labels:list, min_energy:float=0.8):
        """
        Applies PCA to a set of subspaces
        """
        if data.dim() == 1:
            data.unsqueeze_(0)
        set = VectorSet(vector_size=self.vector_size)
        set.populate(data, labels)
        self.subset = set.pca(min_energy=min_energy)

    def eval(self, data:torch.Tensor, correct_labels:list):
        """
        Simple Method that implements a common evaluation procedure for classification problems
        Returns a list of correct classifications, and the success ratio
        """
        if data.dim() == 1:
            data.unsqueeze_(0)
        assert(data.shape[0] == len(correct_labels))
        assert(data.shape[1] == self.vector_size)
        
        predicted_labels = self.classify(data)
        
        correct_class = []
        for l1, l2 in zip(predicted_labels, correct_labels):
            correct_class.append(l1 == l2)
        
        prediction_ratio = correct_class.count(True) / len(correct_class)
        
        return correct_class, prediction_ratio
    
    def classify(self, vectors:torch.Tensor):
        """
        Classifies a tensor in one of the subspaces using the cossine similarity
        """
        if vectors.dim() == 1:
            vectors.unsqueeze_(0)
        assert(vectors.shape[1] == self.vector_size)
        
        # Classify all vectors by cossine similarity
        max_likelihood = [self.subset.labels[0]]*vectors.shape[0]
        cs = [0]*vectors.shape[0]

        for subspace in self.subset:
            foo = self.cossine_similarity(vectors, subspace)
            for i in range(len(foo)):
                if foo[i] > cs[i]: cs[i] = foo[i]; max_likelihood[i] = subspace.label
        return max_likelihood

    def cossine_similarity(self, vector:torch.Tensor, subspace:VectorSpace):
        """
        Returns S = \sum_{i=0}^{r-1} \frac{(x,\phi_i)^2}{\|x\|\|\phi_i\|}
        """
        if vector.dim() == 1:
            vector.unsqueeze_(0)
        assert(vector.shape[1] == subspace.vector_size)
        vector = vector.type(torch.FloatTensor)

        S = torch.sum(
                torch.div(
                    torch.mm(vector, subspace.A.t())**2,
                    torch.matmul(
                        torch.sqrt(
                            torch.diag(
                                torch.mm(vector, vector.t()
                                )
                            )
                        ).unsqueeze(0).t(),
                        torch.sqrt(
                            torch.diag(
                                torch.mm(subspace.A, subspace.A.t())
                            )
                        ).unsqueeze(0)
                    )
                ), dim=1
            )
        return S


# --- unittests
class TestVectorSM(unittest.TestCase):
    def test_init(self):
        pass

    def test_cossine_similarity(self):
        sm = VectorSM()
        subspace = VectorSpace(vector_size=2)
        subspace.append(torch.tensor([1, 0]))
        vector = torch.tensor([0, 1])
        self.assertTrue(torch.allclose(sm.cossine_similarity(vector, subspace), torch.zeros(1)))
        vector = torch.tensor([1, 0])
        self.assertTrue(torch.allclose(sm.cossine_similarity(vector, subspace), torch.ones(1)))
        vector = torch.tensor([[0, 2], [1, 0]])
        sm.cossine_similarity(vector, subspace)

    def test_train(self):
        sm = VectorSM(vector_size=32)
        mock_data = torch.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        sm.train(mock_data, mock_labels)
    
    def test_classify(self):
        sm = VectorSM(vector_size=32)
        mock_data = torch.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        sm.train(mock_data, mock_labels)
        mock_vector = torch.rand(10, 32)
        labels = sm.classify(mock_vector)
        self.assertEqual(len(labels), 10)
    
    def test_eval(self):
        sm = VectorSM(vector_size=32)
        mock_data = torch.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        sm.train(mock_data, mock_labels)
        mock_vector = torch.rand(10, 32)
        mock_labels = [i for i in list(range(10))]
        eval = sm.eval(mock_vector, mock_labels)


if __name__ == "__main__":
    unittest.main()