import json
from typing import List, Union, OrderedDict, Optional

from omegaconf import DictConfig
from hydra.utils import instantiate

import torch


def build_dataset(filepaths: Union[str, List[str]]):
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dataset = []
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as datas:
            dataset += [json.loads(item) for item in datas]

    return dataset


def length_to_mask(length, max_len=None, dtype=None, device=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()

    mask = torch.arange(max_len, device=length.device, dtype=length.dtype)
    mask = mask.expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)

    return mask


def load_module(
    hparams: DictConfig, weights: OrderedDict, device: Optional[str] = "cpu"
) -> torch.nn.Module:
    net = instantiate(hparams)
    net.load_state_dict(weights)
    net.to(device)

    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    return net


def compute_statistics(x, m, dim=-1, eps=1e-12):
    mean = (m * x).sum(dim)
    std = ((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)).sqrt()
    return mean, std
