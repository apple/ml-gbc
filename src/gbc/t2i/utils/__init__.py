# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import sys
import copy
import importlib
import omegaconf
import logging
from typing import Literal
from pathlib import Path
from inspect import isfunction

import torch
import torch.nn as nn

from .loader import load_any


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_class(obj):
    if isinstance(obj, omegaconf.DictConfig):
        obj = dict(**obj)
    if isinstance(obj, dict) and "class" in obj:
        obj_factory = instantiate_class(obj["class"])
        if "factory" in obj:
            obj_factory = getattr(obj_factory, obj["factory"])
        return obj_factory(*obj.get("args", []), **obj.get("kwargs", {}))
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def remove_none(list_x):
    return [i for i in list_x if i is not None]


def balance_sharding_index(total, shards):
    prev = 0
    for i in range(shards):
        this_shard = total // shards
        yield prev, this_shard
        shards -= 1
        total -= this_shard
        prev += this_shard


def balance_sharding(datas, shards):
    total = len(datas)
    for prev, this_shard in balance_sharding_index(total, shards):
        yield datas[prev : prev + this_shard]


def balance_sharding_max_size(datas, max_size):
    total = len(datas)
    shards = total // max_size + int(bool(total % max_size))
    return balance_sharding(datas, shards)


def truncate_or_pad_to_length(
    list_x: list,
    target_length: int,
    padding_mode: Literal["repeat_last", "cycling", "uniform_expansion"],
    deepcopy: bool = False,
):
    if len(list_x) > target_length:
        return list_x[:target_length]
    if len(list_x) == target_length:
        return list_x
    if padding_mode == "repeat_last":
        if deepcopy:
            raise ValueError("deepcopy not supported for padding_mode 'repeat_last'")
        return repeat_last(list_x, target_length)
    if padding_mode == "cycling":
        return cycling(list_x, target_length, deepcopy=deepcopy)
    if padding_mode == "uniform_expansion":
        if deepcopy:
            raise ValueError(
                "deepcopy not supported for padding_mode 'uniform_expansion'"
            )
        return uniform_expansion(list_x, target_length)


def repeat_last(list_x, target_length):
    return list_x + [list_x[-1]] * (target_length - len(list_x))


def cycling(list_x, target_length, deepcopy=False):
    repeats = target_length // len(list_x)
    remainder = target_length % len(list_x)
    if not deepcopy:
        return list_x * repeats + list_x[:remainder]
    # Create deep copies for the repeated part
    repeated_part = [copy.deepcopy(item) for item in list_x for _ in range(repeats)]
    # Create deep copies for the remainder part
    remainder_part = [copy.deepcopy(item) for item in list_x[:remainder]]
    return repeated_part + remainder_part


def uniform_expansion(list_x, target_length):
    result = []
    for idx, ref in enumerate(
        balance_sharding(list(range(target_length)), len(list_x))
    ):
        result.extend([list_x[idx]] * len(ref))
    return result
