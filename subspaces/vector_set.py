import torch
import unittest
from itertools import groupby

from vector_space import VectorSpace


class VectorSet:
    """
    Class that defines a set of subspaces
    """
    def __init__(self, labels:list=[], vector_size:int=1) -> None:
        """
        labels (list): list of labels for each subspace
        vector_size (int): size of vector in each subspace
        """
        self.labels = labels
        self.vector_size = vector_size
        # In this version, we make all subspaces have the same vector size.
        # Maybe this is not completely necessary
        self.set = {label: VectorSpace(vector_size=self.vector_size, label=label) for label in self.labels}
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, label) -> VectorSpace:
        if label in self.labels:
            return self.set[label]
        raise(IndexError(f"Label {label} not in VectorSet"))
    
    def populate(self, vectors: torch.Tensor, labels:list) -> None:
        """
        Populates VectorSpaces with vectors.
        If label does not exists in label list, generate it
        """
        if vectors.dim() == 1:
            vectors.unsqueeze_(0)
        assert(len(vectors) == len(labels))
        assert(vectors.shape[1] == self.vector_size)

        # Order labels and vector
        tensor_list = vectors.tolist()
        lt = zip(labels, tensor_list)
        lt = sorted(lt)
        sorted_tensor = torch.tensor([i for _, i in lt])
        sorted_labels = [i for i, _ in lt]

        # Group labels
        group_list = [list(g) for _, g in groupby(sorted_labels)]

        # Populate subspaces in batches
        i = 0
        for group in group_list:
            label = group[0]
            if label not in self.labels:
                self.labels.append(label)
                self.set[label] = VectorSpace(vector_size=self.vector_size, label=label)

            self.set[label].append(sorted_tensor[i:i+len(group)])
            i += len(group)

        # for vector, label in zip(vectors, labels):
        #     # Check if label exists. If not, generate new subspace
        #     if label not in self.labels:
        #         self.labels.append(label)
        #         self.set[label] = VectorSpace(vector_size=self.vector_size, label=label)
        #     self.set[label].append(vector)
    
    def pca(self, min_energy:float=0.8):
        subset = VectorSet(labels=self.labels, vector_size=self.vector_size)
        for label, subspace in self.set.items():
            subsubspace = subspace.pca(min_energy)
            subset.populate(subsubspace.A, [label]*len(subsubspace))
        return subset


# --- unittests
class TestSubspaces(unittest.TestCase):
    def test_init(self):
        _ = VectorSet()
    
    def test_getitem_len(self):
        set = VectorSet(list(range(10)))
        self.assertEqual(len(set), 10)
        with self.assertRaises(IndexError):
            subspace = set[11]
        subspace = set[9]
    
    def test_populate(self):
        set = VectorSet(vector_size=32)

        mock_data = torch.rand(10, 32)
        mock_labels = list(range(10))

        for _ in range(10):
            set.populate(mock_data, mock_labels)
        
        assert(len(set.labels) == 10)
        assert(len(set[0]) == 10)


        with self.assertRaises(AssertionError):
            mock_data = torch.rand(10, 32)
            mock_labels = list(range(9))
            set.populate(mock_data, mock_labels)

        with self.assertRaises(AssertionError):
            mock_data = torch.rand(10, 24)
            mock_labels = list(range(10))
            set.populate(mock_data, mock_labels)
    
    def test_pca(self):
        set = VectorSet(vector_size=32)
        mock_data = torch.rand(100, 32)
        mock_labels = [i%10 for i in list(range(100))]
        set.populate(mock_data, mock_labels)
        subset = set.pca()

if __name__ == "__main__":
    unittest.main()