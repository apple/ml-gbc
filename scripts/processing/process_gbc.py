# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

from gbc.utils import setup_gbc_logger
from gbc.data import GbcGraphFull
from gbc.processing import local_process_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JSONL/JSON/Parquet files to JSON/JSONL/Parquet."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        default=None,
        help="List of input files or directories.",
    )
    parser.add_argument(
        "--input_formats",
        nargs="+",
        default=None,
        help="List of input formats to look for (e.g., .json, .jsonl, .parquet).",
    )
    parser.add_argument(
        "--save_format",
        default=None,
        help="Desired output format (e.g., .json, .jsonl, .parquet).",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Directory to save the converted files.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=None,
        help="List of configs to be used. Latter ones override former ones.",
    )
    args = parser.parse_args()

    if args.configs is not None:
        configs = []
        for config in args.configs:
            conf = OmegaConf.load(config)
            configs.append(conf)
        config = OmegaConf.merge(*configs)
    else:
        config = OmegaConf.create()
    if "processing_config" in config:
        config = config.processing_config
    config = instantiate(config)

    for key, value in vars(args).items():
        if key == "configs":
            continue
        if value is not None:
            config[key] = value
        assert key in config, f"{key} not found in neither args nor config"

    setup_gbc_logger()
    os.makedirs(config.save_dir, exist_ok=True)

    data_transform = config.get("data_transform", None)
    name_transform = config.get("name_transform", None)

    local_process_data(
        config.input_paths,
        save_dir=config.save_dir,
        save_format=config.save_format,
        input_formats=config.input_formats,
        data_class=GbcGraphFull,
        data_transform=data_transform,
        name_transform=name_transform,
    )
