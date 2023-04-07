import torch
import unittest
from itertools import groupby
from typing import List

from vector_space import VectorSpace
from vector_set import VectorSet
from vector_sm import VectorSM

class VectorMSM(VectorSM):
    """
    Class that implements a simple mutual subspace method for vector subspaces
    """
    # def eval(self, data:list(VectorSpace), correct_labels:list):
    #     """
    #     Simple Method that implements a common evaluation procedure for classification problems
    #     Returns a list of correct classifications, and the success ratio
    #     """
    #     if data.dim() == 1:
    #         data.unsqueeze_(0)
    #     assert(data.shape[0] == len(correct_labels))
    #     assert(data.shape[1] == self.vector_size)
        
    #     predicted_labels = self.classify(data)
        
    #     correct_class = []
    #     for l1, l2 in zip(predicted_labels, correct_labels):
    #         correct_class.append(l1 == l2)
        
    #     prediction_ratio = correct_class.count(True) / len(correct_class)
        
    #     return correct_class, prediction_ratio
    
    # def classify(self, vspace:list(VectorSpace)):
    #     """
    #     Classifies a tensor in one of the subspaces using the cossine similarity
    #     """
    #     if vectors.dim() == 1:
    #         vectors.unsqueeze_(0)
    #     assert(vectors.shape[1] == self.vector_size)
        
    #     # Classify all vectors by cossine similarity
    #     max_likelihood = [self.subset.labels[0]]*vectors.shape[0]
    #     cs = [0]*vectors.shape[0]

    #     for subspace in self.subset:
    #         foo = self.cossine_similarity(vectors, subspace)
    #         for i in range(len(foo)):
    #             if foo[i] > cs[i]: cs[i] = foo[i]; max_likelihood[i] = subspace.label
    #     return max_likelihood

    def cossine_similarity(self, vspaces: List[VectorSpace], subspace:VectorSpace):
        """
        Returns the cossine similarity between a list of vector spaces and one subspace
        """
        # Order list of vector spaces and group vector spaces of same size in batches
        original_order = torch.linspace(1, len(vspaces), len(vspaces)).tolist()

        nlist = [space.n for space in vspaces]
        z = zip(vspaces, nlist, original_order)
        z = sorted(z)

        sorted_vspaces = []
        sorted_nlist = []
        new_order = []

        for vs, nl, no in z:
            sorted_vspaces.append(vs)
            sorted_nlist.append(nl)
            new_order.append(no)

        group_nlist = [len(list(n)) for _, n in groupby(sorted_nlist)]

        # Group vector spaces in 3d tensors, calculate cossine similarities in batches
        i = 0
        cossine_similarities = []
        for n in group_nlist:
            vspace_tensor = torch.vstack([vspace.A.unsqueeze(0) for vspace in sorted_vspaces[i:i+n]])
            # vspace_tensor.shape = [vspace, vspace.n, vspace.vector_size]
            batch_subspace = subspace.A.unsqueeze(0).repeat(n, 1, 1)

            X = torch.bmm(
                    torch.bmm(batch_subspace, vspace_tensor.transpose(1, 2)),
                    torch.bmm(batch_subspace, vspace_tensor.transpose(1, 2)).transpose(1, 2)
            )

            L, Q = torch.linalg.eigh(X) # X is hermitian, can use eigh
            cossine_similarities.append(L)
            i += n

        cossine_similarities = torch.vstack(cossine_similarities)

        # Unsort cossine similarities
        z = zip(new_order, cossine_similarities)
        z = sorted(z)
        cossine_similarities = torch.vstack([space for _, space in z])

        return cossine_similarities


# --- unittests
class TestVectorSM(unittest.TestCase):
    def test_init(self):
        pass

    def test_cossine_similarity(self):
        msm = VectorMSM()
        subspace = VectorSpace(vector_size=2)
        subspace.append(torch.tensor([[0, 1], [1, 0]]))
        subspace.append(torch.tensor([[0, 1], [1, 0]]))
        vspace12 = VectorSpace(vector_size=2)
        vspace12.append(torch.tensor([[1, 1]]))
        vspace22 = VectorSpace(vector_size=2)
        vspace22.append(torch.tensor([[-1, 1], [1, -1]]))
        vspace32 = VectorSpace(vector_size=2)
        vspace32.append(torch.tensor([[-1, 1], [1, -1], [0, 1]]))

        test_list = [vspace22, vspace32, vspace22, vspace32, vspace12, vspace12, vspace12]

        similarity = msm.cossine_similarity(test_list, subspace)


    # def test_train(self):
    #     sm = VectorSM(vector_size=32)
    #     mock_data = torch.rand(100, 32)
    #     mock_labels = [i%10 for i in list(range(100))]
    #     sm.train(mock_data, mock_labels)
    
    # def test_classify(self):
    #     sm = VectorSM(vector_size=32)
    #     mock_data = torch.rand(100, 32)
    #     mock_labels = [i%10 for i in list(range(100))]
    #     sm.train(mock_data, mock_labels)
    #     mock_vector = torch.rand(10, 32)
    #     labels = sm.classify(mock_vector)
    #     self.assertEqual(len(labels), 10)
    
    # def test_eval(self):
    #     sm = VectorSM(vector_size=32)
    #     mock_data = torch.rand(100, 32)
    #     mock_labels = [i%10 for i in list(range(100))]
    #     sm.train(mock_data, mock_labels)
    #     mock_vector = torch.rand(10, 32)
    #     mock_labels = [i for i in list(range(10))]
    #     eval = sm.eval(mock_vector, mock_labels)


if __name__ == "__main__":
    unittest.main()