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
    
    def classify(self, vspaces: List[VectorSpace]):
        """
        Classifies a list of subspace in one of the subspaces using the subspace_similarity metric
        """
        # Check if passing a list or a VectorSpace. Correct if necessary
        if type(vspaces) is not list:
            vspaces = [vspaces]
        assert(type(vspaces[0] == VectorSpace))

        # Classify all vectors by subspace similarity
        max_likelihood = [self.subset.labels[0]]*len(vspaces)
        ss = [0]*len(vspaces)

        for subspace in self.subset:
            foo = self.subspace_similarity(vspaces, subspace)
            for i in range(len(foo)):
                if foo[i] > ss[i]: ss[i] = foo[i]; max_likelihood[i] = subspace.label
            
        return max_likelihood

    def subspace_similarity(self, vspaces: List[VectorSpace], subspace:VectorSpace):
        cossine_similarities = self.cossine_similarity(vspaces, subspace)
        S = torch.sum(cossine_similarities, dim=1) / cossine_similarities.shape[1]
        return S
        
    def cossine_similarity(self, vspaces: List[VectorSpace], subspace:VectorSpace):
        """
        Returns the cossine similarity between a list of vector spaces and one subspace
        return shape: [vspace, cos^2_i]
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

        # Unsort axis=0 cossine similarities. Sort axis=1 cossine_similarities by descending order
        z = zip(new_order, cossine_similarities)
        z = sorted(z)
        cossine_similarities = torch.vstack([space for _, space in z])
        cossine_similarities, _ = cossine_similarities.sort(dim=1, descending=True)

        return cossine_similarities


# --- unittests
class TestVectorSM(unittest.TestCase):
    def test_init(self):
        pass

    def test_cossine_similarity(self):
        msm = VectorMSM()
        subspace = VectorSpace(vector_size=3)
        subspace.append(torch.tensor([[0, 0, 1], [0, 1, 0]]))
        vspace12 = VectorSpace(vector_size=3)
        vspace12.append(torch.tensor([[1, 0, 0]]))
        vspace22 = VectorSpace(vector_size=3)
        vspace22.append(torch.tensor([[1, 0, 0], [0, 0, 1]]))
        vspace32 = VectorSpace(vector_size=3)
        vspace32.append(torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, -1]]))

        vspace_list = [vspace22, vspace32, vspace22, vspace32, vspace12, vspace12, vspace12]

        similarity = msm.cossine_similarity(vspace_list, subspace)
        self.assertTrue(torch.allclose(similarity[4], torch.zeros(2)))
        self.assertTrue(torch.allclose(similarity[5], torch.zeros(2)))
        self.assertTrue(torch.allclose(similarity[6], torch.zeros(2)))
        self.assertTrue(torch.allclose(similarity[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(similarity[2], torch.tensor([1.0, 0.0])))
    
    def test_subspace_similarities(self):
        msm = VectorMSM()
        subspace = VectorSpace(vector_size=3)
        subspace.append(torch.tensor([[0, 0, 1], [0, 1, 0]]))
        vspace12 = VectorSpace(vector_size=3)
        vspace12.append(torch.tensor([[1, 0, 0]]))
        vspace22 = VectorSpace(vector_size=3)
        vspace22.append(torch.tensor([[1, 0, 0], [0, 0, 1]]))

        vspace_list = [vspace22, vspace12, vspace22, vspace12, vspace12, vspace22]

        self.assertTrue(torch.allclose(msm.subspace_similarity(vspace_list, subspace), torch.tensor([0.5000, 0.0000, 0.5000, 0.0000, 0.0000, 0.5000])))

    def test_train(self):
        msm = VectorMSM(vector_size=32)
        mock_data = torch.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        msm.train(mock_data, mock_labels)
    
    def test_classify(self):
        msm = VectorMSM(vector_size=32)
        mock_data = torch.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        msm.train(mock_data, mock_labels)

        mock_subspace1 = VectorSpace(vector_size=32)
        mock_subspace1.append(torch.rand(6, 32))
        mock_subspace2 = VectorSpace(vector_size=32)
        mock_subspace2.append(torch.rand(5, 32))
        mock_subspace3 = VectorSpace(vector_size=32)
        mock_subspace3.append(torch.rand(7, 32))

        labels = msm.classify(mock_subspace1)
        self.assertEqual(len(labels), 1)

        labels = msm.classify([mock_subspace1, mock_subspace2, mock_subspace3])
        self.assertEqual(len(labels), 3)


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