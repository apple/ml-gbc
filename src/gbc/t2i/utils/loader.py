# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Any
import omegaconf
from hydra.utils import instantiate
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelLoadingConfig:
    ckpt_path: str | None = None
    state_dict_key: str | None = None
    state_dict_prefix: str | None = None
    precision: str | None = None
    device: str | None = None
    to_compile: bool = False
    to_freeze: bool = False


def load_torch_model_from_path(path: str):
    if path.endswith(".safetensors"):
        import safetensors

        return safetensors.torch.load_file(path, device="cpu")
    return torch.load(path, map_location=lambda storage, loc: storage)


def extract_state_dict(state_dict: dict[str, Any], key: str | None, prefix: str | None):
    if key is not None:
        state_dict = state_dict[key]
    if prefix is None:
        return state_dict
    extracted_state_dict = {}
    for key, params in state_dict.items():
        if key.startswith(prefix):
            extracted_state_dict[key[len(prefix) :]] = params
    return extracted_state_dict


def prepare_model(model: nn.Module, model_loading_config: ModelLoadingConfig):
    if model_loading_config.ckpt_path is not None:
        state_dict_all = load_torch_model_from_path(model_loading_config.ckpt_path)
        state_dict = extract_state_dict(
            state_dict_all,
            model_loading_config.state_dict_key,
            model_loading_config.state_dict_prefix,
        )
        model.load_state_dict(state_dict)
    if model_loading_config.precision is not None:
        model = model.to(eval(model_loading_config.precision))
    if model_loading_config.device is not None:
        model = model.to(model_loading_config.device)
    if model_loading_config.to_compile:
        model = torch.compile(model)
    if model_loading_config.to_freeze:
        model.requires_grad_(False).eval()
    return model


def load_any(obj):
    if isinstance(obj, dict) or isinstance(obj, omegaconf.DictConfig):
        if "_load_config_" in obj:
            load_config = obj.pop("_load_config_")
            load_config = ModelLoadingConfig(**load_config)
        else:
            load_config = None
        obj = instantiate(obj)
        if load_config is not None:
            obj = prepare_model(obj, load_config)
    return obj
