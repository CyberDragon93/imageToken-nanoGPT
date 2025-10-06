import torch
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
from functools import partial
from typing import Callable, Any


class CouplingDataset(Dataset):
    def __init__(
        self,
        D1: torch.utils.data.Dataset | torch.Tensor | dist.Distribution | Callable,
        D0: (
            torch.utils.data.Dataset
            | torch.Tensor
            | dist.Distribution
            | Callable
            | None
        ) = None,
        reflow: bool = False,
    ):
        """
        A dataset that provides coupled samples from D0 and D1.

        Args:
            D1: Dataset, Tensor, distribution, or callable.
                The target data samples.
                All should return data after applying transforms, such as ToTensor().
            D0: None (default), Dataset, Tensor, distribution, or callable.
                The noise data samples. If None and reflow=False, defaults to standard normal.
            reflow: bool.
                If True, D0 and D1 are paired samples.
                If False, D0 is generated independently.
        """
        self.D1 = D1
        self.D0 = D0
        self.reflow = reflow

        self.D1_mode = self._get_D_mode(self.D1)

        if self.D0 is None and not self.reflow:
            self.D0 = self._set_default_noise()
            self.D0_mode = "distribution"
        else:
            self.D0_mode = self._get_D_mode(self.D0)

        self.length = self._determine_length()

        self._validate_inputs()

    def _get_D_mode(self, D):
        if D is None:
            return "none"
        elif isinstance(D, torch.Tensor):
            return "tensor"
        elif isinstance(D, torch.utils.data.Dataset):
            return "dataset"
        elif isinstance(D, torch.distributions.Distribution):
            return "distribution"
        elif callable(D):
            return "callable"
        else:
            raise NotImplementedError(f"Unsupported type: {type(D)}")

    def _set_default_noise(self):
        # Need to know sample shape
        sample_shape = self._get_sample_shape(self.D1)
        return dist.Normal(torch.zeros(sample_shape), torch.ones(sample_shape))

    def _get_sample_shape(self, D):
        sample = self._get_sample(D, 0, self._get_D_mode(D))
        if isinstance(sample, torch.Tensor):
            return sample.shape
        elif isinstance(sample, (tuple, list)):
            # Assume the first element is the data
            return sample[0].shape
        elif isinstance(sample, dict):
            # Assume there is a key 'data' or take the first item
            return next(iter(sample.values())).shape
        else:
            raise NotImplementedError(
                f"Cannot determine sample shape for type: {type(sample)}"
            )

    def _determine_length(self):
        if self.reflow:
            len_D1 = self._get_length(self.D1)
            len_D0 = self._get_length(self.D0)
            assert (
                len_D1 == len_D0
            ), "D0 and D1 must have the same length when reflow=True"
            return len_D1
        else:
            return self._get_length(self.D1)

    def _get_length(self, D):
        if isinstance(D, torch.Tensor) or isinstance(D, torch.utils.data.Dataset):
            return len(D)
        else:
            return self.default_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        X1 = self._get_sample(self.D1, index, self.D1_mode)
        if self.reflow:
            X0 = self._get_sample(self.D0, index, self.D0_mode)
        else:
            if self.D0_mode == "distribution":
                X0 = None  # Will be handled in collate_fn
            else:
                X0 = self._get_sample(self.D0, index, self.D0_mode)
        return X0, X1

    def _get_sample(self, D, index, mode):
        if mode == "tensor":
            return D[index]
        elif mode == "dataset":
            sample = D[index]
            return sample  # Return full sample (could be tuple or dict)
        elif mode == "distribution":
            # Return None, will be handled in collate_fn
            return None
        elif mode == "callable":
            return D()
        elif mode == "none":
            return None
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    def _validate_inputs(self):
        if self.reflow:
            # D0 and D1 must be datasets or tensors of same length
            assert self.D0_mode in [
                "tensor",
                "dataset",
            ], "When reflow=True, D0 must be a tensor or dataset"
            assert self.D1_mode in [
                "tensor",
                "dataset",
            ], "When reflow=True, D1 must be a tensor or dataset"
            len_D0 = self._get_length(self.D0)
            len_D1 = self._get_length(self.D1)
            assert (
                len_D0 == len_D1
            ), "D0 and D1 must have the same length when reflow=True"
        else:
            # D1 must be provided
            assert self.D1 is not None, "D1 must be provided"

    default_length = 10000


# Custom collate function
def coupling_collate_fn(batch, D0_distribution=None):
    X0_list, X1_list = zip(*batch)
    # Handle X0_list
    if all(x is None for x in X0_list) and D0_distribution is not None:
        batch_size = len(X0_list)
        # Sample batch from D0 distribution
        X0_batch = D0_distribution.sample([batch_size])
    else:
        X0_batch = collate_batch_elements(X0_list)

    # Handle X1_list
    X1_batch = collate_batch_elements(X1_list)

    return X0_batch, X1_batch


def collate_batch_elements(batch_elements):
    if isinstance(batch_elements[0], torch.Tensor):
        return torch.stack(batch_elements)
    elif isinstance(batch_elements[0], (tuple, list)):
        # Recursively collate each element in the tuple/list
        transposed = list(zip(*batch_elements))
        return [collate_batch_elements(elements) for elements in transposed]
    elif isinstance(batch_elements[0], dict):
        # Recursively collate each value in the dict
        collated = {}
        for key in batch_elements[0]:
            collated[key] = collate_batch_elements([d[key] for d in batch_elements])
        return collated
    else:
        # If it's a scalar or unstackable type, return as a list
        return batch_elements
