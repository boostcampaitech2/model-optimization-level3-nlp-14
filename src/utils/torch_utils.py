"""Common utility functions.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch._utils import _accumulate
from torch import randperm, default_generator

from sklearn.model_selection import train_test_split

def convert_model_to_torchscript(
    model: nn.Module, path: Optional[str] = None
) -> torch.jit.ScriptModule:
    """Convert PyTorch Module to TorchScript.

    Args:
        model: PyTorch Module.

    Return:
        TorchScript module.
    """
    model.eval()
    jit_model = torch.jit.script(model)

    if path:
        jit_model.save(path)

    return jit_model


def stratified_random_split(dataset, lengths):
    r"""
    Stratified Version of torch.utils.data.random_split
    randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> stratified_random_split(range(10), [3, 7])

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    val_ratio = lengths[1] / len(dataset)
    train_indices, val_indices = train_test_split(list(range(len(dataset.targets))), test_size=val_ratio, stratify=dataset.targets)
    return [Subset(dataset, train_indices), Subset(dataset, val_indices)]


def split_dataset_index(
    train_dataset: torch.utils.data.Dataset, n_data: int, split_ratio: float = 0.1
) -> Tuple[Subset, Subset]:
    """Split dataset indices with split_ratio.

    Args:
        n_data: number of total data
        split_ratio: split ratio (0.0 ~ 1.0)

    Returns:
        SubsetRandomSampler ({split_ratio} ~ 1.0)
        SubsetRandomSampler (0 ~ {split_ratio})
    """
    indices = np.arange(n_data)
    split = int(split_ratio * indices.shape[0])

    train_idx = indices[split:]
    valid_idx = indices[:split]

    train_subset = Subset(train_dataset, train_idx)
    valid_subset = Subset(train_dataset, valid_idx)

    return train_subset, valid_subset


def subset_sampler(dataset: torch.utils.data.Dataset, sampling_ratio: int=0.1):
    """Make subset by sampling from dataset

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be sampled
        sampling_ratio (int, optional): Defaults to 0.1.

    Returns:
        torch.utils.data.Dataset: subset of dataset
    """
    _, subset_indices = train_test_split(list(range(len(dataset.targets))), test_size=sampling_ratio, stratify=dataset.targets)
    return Subset(dataset, subset_indices)


def save_model(model, path, optmizer, scheduler):
    """save model to torch script, onnx."""
    try:
        torch.save(model.state_dict(), f=path)
        ts_path = os.path.splitext(path)[:-1][0] + ".ts"
        scheduler_optimizer_path = os.path.splitext(path)[:-1][0] + f"_sch_opt_step{scheduler.last_epoch}.pth.tar"
        torch.save({
            'optimizer' : optmizer.state_dict(),
            'scheduler' : scheduler.state_dict()
            },
            scheduler_optimizer_path)
        convert_model_to_torchscript(model, ts_path)
    except Exception:
        print("Failed to save torch")


def model_info(model, verbose=False):
    """Print out model info."""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma")
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    print(
        f"Model Summary: {len(list(model.modules()))} layers, "
        f"{n_p:,d} parameters, {n_g:,d} gradients"
    )


@torch.no_grad()
def check_runtime(
    model: nn.Module, img_size: List[int], device: torch.device, repeat: int = 100
) -> float:
    repeat = min(repeat, 20)
    img_tensor = torch.rand([1, *img_size]).to(device)
    measure = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model.eval()
    for _ in range(repeat):
        start.record()
        # _ = model(img_tensor)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        measure.append(start.elapsed_time(end))

    measure.sort()
    n = len(measure)
    k = int(round(n * (0.2) / 2))
    trimmed_measure = measure[k + 1 : n - k]

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        # _ = model(img_tensor)
        pass
    print(prof)
    print("measured time(ms)", np.mean(trimmed_measure))
    model.train()
    return np.mean(trimmed_measure)


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def autopad(
    kernel_size: Union[int, List[int]], padding: Union[int, None] = None
) -> Union[int, List[int]]:
    """Auto padding calculation for pad='same' in TensorFlow."""
    # Pad to 'same'
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    return padding or [x // 2 for x in kernel_size]


class Activation:
    """Convert string activation name to the activation class."""

    def __init__(self, act_type: Union[str, None]) -> None:
        """Convert string activation name to the activation class.

        Args:
            type: Activation name.

        Returns:
            nn.Identity if {type} is None.
        """
        self.type = act_type
        self.args = [1] if self.type == "Softmax" else []

    def __call__(self) -> nn.Module:
        if self.type is None:
            return nn.Identity()
        elif hasattr(nn, self.type):
            return getattr(nn, self.type)(*self.args)
        else:
            return getattr(
                __import__("src.modules.activations", fromlist=[""]), self.type
            )()


class EarlyStopping:
    """????????? patience ????????? validation loss??? ???????????? ????????? ????????? ?????? ??????
    Ref : https://quokkas.tistory.com/37
    """
    def __init__(self, path, patience=7, verbose=False, delta=0):
        """
        Args:
            path (str): checkpoint?????? ??????
            patience (int): validation loss??? ????????? ??? ???????????? ??????
                            Default: 7
            verbose (bool): True??? ?????? ??? validation loss??? ?????? ?????? ????????? ??????
                            Default: False
            delta (float): ?????????????????? ???????????? monitered quantity??? ?????? ??????
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, optmizer, scheduler):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optmizer, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optmizer, scheduler)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optmizer, scheduler):
        '''validation loss??? ???????????? ????????? ????????????.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, self.path, optmizer, scheduler)
        self.val_loss_min = val_loss

if __name__ == "__main__":
    # test
    check_runtime(None, [32, 32] + [3])
