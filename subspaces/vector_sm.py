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

    def train(self, data:torch.Tensor, labels:list, min_energy:float=0.8):
        """
        Applies PCA to a set of subspaces
        """
        if data.dim() == 1:
            data.unsqueeze_(0)
        set = VectorSet(vector_size=self.vector_size)
        set.Populate(data, labels)
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
        max_likelihood = []
        for vector in vectors:
            cs = 0
            for subspace in self.subset:
                foo = self.cossine_similarity(vector, subspace)
                if foo > cs: cs = foo; label = subspace.label
            max_likelihood.append(label)
        return max_likelihood

    def cossine_similarity(self, vector:torch.Tensor, subspace:VectorSubspace):
        """
        Returns S = \sum_{i=0}^{r-1} \frac{(x,\phi_i)^2}{\|x\|\|\phi_i\|}
        """
        assert(len(vector) == subspace.vector_size)
        vector = vector.type(torch.FloatTensor)
        S = 0
        for i in range(len(subspace)): # You can speed this up
            S+=torch.inner(vector, subspace[i])**2 / (torch.inner(vector, vector) * torch.inner(subspace[i], subspace[i]))
        return S

# --- unittests
class TestVectorSM(unittest.TestCase):
    def test_init(self):
        pass

    def test_cossine_similarity(self):
        sm = VectorSM()
        subspace = VectorSubspace(vector_size=2)
        subspace.append(torch.tensor([1, 0]))
        vector = torch.tensor([0, 1])
        self.assertTrue(torch.allclose(sm.cossine_similarity(vector, subspace), torch.zeros(1)))
        vector = torch.tensor([1, 0])
        self.assertTrue(torch.allclose(sm.cossine_similarity(vector, subspace), torch.ones(1)))
    
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
        print(eval)


if __name__ == "__main__":
    unittest.main()