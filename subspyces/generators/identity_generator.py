import torch
from torch.utils.data import Dataset, DataLoader
from itertools import groupby
from typing import Dict, Any

from subspyces.core import VectorSpace
from . import AbstractGenerator


class IdentityGenerator(AbstractGenerator):
    """
    Takes a torch dataset and organizes it into many VectorSpace, one for each label.
    It's called identity because it doesn't modify the input in any way, just organizing
    it in vector spaces.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return "IdentityGenerator"

    def generate(self, dataset: Dataset,
                 batch_size: int = 32, *args, **kwargs) -> Dict[Any, VectorSpace]:
        """
        Populates VectorSpace with vectors.
        If label does not exists in label list, generate it
        """
        loader = DataLoader(dataset, batch_size=batch_size)
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

            # If labels are torch.Tensor(.), transform then to integers
            # torch.Tensors *especifically* are giving problems on dict indexing.
            if type(lt[0][0]) is torch.Tensor:
                sorted_labels = [int(i) for i, _ in lt]
            else:
                sorted_labels = [i for i, _ in lt]

            # Group labels
            group_list = [list(g) for _, g in groupby(sorted_labels)]

            # Populate subspyces in minibatches
            i = 0
            for group in group_list:
                label = group[0]
                # Create subspace if it doesn't exist
                if label not in v_space_list:
                    v_space_list[label] = VectorSpace(dim=batch_data.shape[1], label=label)
                v_space_list[label].append(sorted_tensor[i:i+len(group)])
                i += len(group)

        return v_space_list
