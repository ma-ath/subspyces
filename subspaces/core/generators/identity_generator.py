import torch
from torch.utils.data import Dataset, DataLoader
from itertools import groupby

from subspaces import VectorSet, VectorSpace
from .abstract_generator import AbstractGenerator


class IdentityGenerator(AbstractGenerator):
    """
    Takes a set of data and generates a VectorSet where each VectorSpace is simply the
    input data organized by labels
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return "IdentityGenerator"

    def generate(self, dataset: Dataset, batch_size: int = 32, *args, **kwargs) -> VectorSet:
        """
        Populates VectorSpaces with vectors.
        If label does not exists in label list, generate it
        """
        loader = DataLoader(dataset, batch_size=batch_size)
        label_list = []
        v_space_list = {}

        for data, labels in next(iter(loader)):
            # Format data
            if data.dim() == 1:
                data.unsqueeze_(0)

            # Order labels and vector
            tensor_list = data.tolist()
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
                if label not in label_list:
                    label_list.append(label)
                    v_space_list[label] = VectorSpace(dim=data.shape[0], label=label)

                self.set[label].append(sorted_tensor[i:i+len(group)])
                i += len(group)

            # for vector, label in zip(vectors, labels):
            #     # Check if label exists. If not, generate new vspace
            #     if label not in self.labels:
            #         self.labels.append(label)
            #         self.set[label] = VectorSpace(vector_size=self.vector_size, label=label)
            #     self.set[label].append(vector)