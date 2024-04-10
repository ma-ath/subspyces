import torch
from torch.utils.data import Dataset, DataLoader
from itertools import groupby
from typing import List

from subspaces import VectorSpace
from .abstract_generator import AbstractGenerator


class IdentityGenerator(AbstractGenerator):
    """
    Takes a torch dataset and organizes it into many VectorSpace, one for each label
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return "IdentityGenerator"

    def generate(self, dataset: Dataset, batch_size: int = 32, *args, **kwargs) -> List[VectorSpace]:
        """
        Populates VectorSpace with vectors.
        If label does not exists in label list, generate it
        """
        loader = DataLoader(dataset, batch_size=batch_size)
        label_list = []
        v_space_list = dict()

        for batch_data, batch_label in loader:
            # Format data
            if batch_data.dim() == 1:
                batch_data.unsqueeze_(0)
            if batch_data.dim() > 2:
                raise (
                    RuntimeError(f"Data must be a Vector, but it has dimension {batch_data.dim()}"))

            # Order labels and vector
            tensor_list = batch_data.tolist()
            lt = zip(batch_label, tensor_list)
            lt = sorted(lt)
            sorted_tensor = torch.tensor([i for _, i in lt])
            sorted_labels = [i for i, _ in lt]

            # Group labels
            group_list = [list(g) for _, g in groupby(sorted_labels)]

            # Populate subspaces in minibatches
            i = 0
            for group in group_list:
                label = group[0]
                # Create subspace if it doesn't exist
                if label not in label_list:
                    label_list.append(label)
                    v_space_list[label] = VectorSpace(dim=batch_data.shape[1], label=label)
                v_space_list[label].append(sorted_tensor[i:i+len(group)])
                i += len(group)

        return v_space_list.items()
