import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Distribution, Normal
from typing import Optional, List, Dict, Union
from functools import partial


class CouplingDataset(Dataset):
    def __init__(
        self,
        D1: torch.Tensor | Distribution,
        D0: torch.Tensor | Distribution | None = None,
        reflow: bool = False,
        condition: List[Dict] | List | None = None,
    ):
        """
        A simplified coupling dataset that provides coupled samples from D0 and D1.

        Args:
            D1: Tensor or Distribution.
                The target data samples of shape [B, D...].
            D0: None (default), Tensor, or Distribution.
                The noise data samples of shape [B, D...]. If None and reflow=False, defaults to standard normal.
            reflow: bool.
                If True, D0 and D1 are paired tensors.
                If False, D0 is generated independently.
            condition: Optional[List[Dict]].
                Additional conditional information for D1. List of dicts, one per sample.
        """
        self.D1 = D1
        self.D0 = D0
        self.reflow = reflow
        self.condition = condition

        # Determine modes for D1 and D0
        self.D1_mode = "tensor" if isinstance(D1, torch.Tensor) else "distribution"

        if self.D0 is None and not self.reflow:
            self.D0 = self._set_default_noise()
            self.D0_mode = "distribution"
        else:
            self.D0_mode = (
                "tensor" if isinstance(self.D0, torch.Tensor) else "distribution"
            )

        self.length = self._determine_length()

        self._validate_inputs()

    def _set_default_noise(self):
        sample_shape = self._get_sample_shape(self.D1)
        return Normal(torch.zeros(sample_shape), torch.ones(sample_shape))

    def _get_sample_shape(self, D):
        if isinstance(D, torch.Tensor):
            return D.shape[1:]  # Exclude batch dimension
        elif isinstance(D, Distribution):
            sample = D.sample()
            return sample.shape
        else:
            raise ValueError("D must be a tensor or distribution")

    def _determine_length(self):
        if self.reflow:
            assert isinstance(self.D0, torch.Tensor) and isinstance(
                self.D1, torch.Tensor
            ), "When reflow=True, D0 and D1 must be tensors"
            len_D0 = len(self.D0)
            len_D1 = len(self.D1)
            assert (
                len_D0 == len_D1
            ), "D0 and D1 must have the same length when reflow=True"
            return len_D0
        else:
            if isinstance(self.D1, torch.Tensor):
                return len(self.D1)
            else:
                return self.default_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Get X1
        X1 = self._get_sample(self.D1, index, self.D1_mode)
        # Get condition if provided
        condition = self.condition[index] if self.condition is not None else None

        if self.reflow:
            X0 = self._get_sample(self.D0, index, self.D0_mode)
        else:
            if self.D0_mode == "distribution":
                X0 = None  # Will be handled in collate_fn
            else:
                X0 = self._get_sample(self.D0, index, self.D0_mode)
        return X0, X1, condition

    def _get_sample(self, D, index, mode):
        if mode == "tensor":
            return D[index]
        elif mode == "distribution":
            # Return None, sample in collate_fn
            return None
        else:
            raise ValueError("Unsupported mode")

    def _validate_inputs(self):
        if self.reflow:
            assert isinstance(self.D0, torch.Tensor) and isinstance(
                self.D1, torch.Tensor
            ), "When reflow=True, D0 and D1 must be tensors"
            if self.condition is not None:
                assert (
                    len(self.condition) == self.length
                ), "Length of condition must match length of D1"
        else:
            if isinstance(self.D1, torch.Tensor):
                if self.condition is not None:
                    assert len(self.condition) == len(
                        self.D1
                    ), "Length of condition must match length of D1"

    default_length = 10000


# Custom collate function
def coupling_collate_fn(batch, D0_distribution=None, D1_distribution=None):
    X0_list, X1_list, condition_list = zip(*batch)
    # Handle X0
    if all(x is None for x in X0_list) and D0_distribution is not None:
        batch_size = len(X0_list)
        X0_batch = D0_distribution.sample([batch_size])
    else:
        X0_batch = torch.stack(X0_list)
    # Handle X1
    if all(x is None for x in X1_list) and D1_distribution is not None:
        batch_size = len(X1_list)
        X1_batch = D1_distribution.sample([batch_size])
    else:
        X1_batch = torch.stack(X1_list)
    # Handle condition
    if condition_list[0] is not None:
        condition_batch = condition_list
    else:
        condition_batch = None
    return X0_batch, X1_batch, condition_batch
