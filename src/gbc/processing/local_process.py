# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
from pathlib import Path
from typing import Callable, Optional, Type

from gbc.utils import load_list_from_file, save_list_to_file, get_files_recursively
from .meta_process import meta_process_data


def local_process_data(
    inputs: list[str],
    save_dir: str,
    save_format: str,
    *,
    input_formats: Optional[list[str]] = None,
    data_class: Optional[Type] = None,
    data_transform: Optional[Callable] = None,
    name_transform: Optional[Callable[[str], str]] = None,
):
    """
    Processes local data files, optionally transforming and combining them,
    and saves the results.

    This function handles data processing for a list of input paths, where
    each path can be a file or a directory.
    It supports loading, transforming, and combining data, then saving
    the results to a specified directory in a specified format.
    Both the loaded data and the saved data list of dictionaries.

    Parameters
    ----------
    inputs
        List of file or directory paths to be processed.
    save_dir
        Directory where the processed files will be saved.
    save_format
        Format in which to save the processed files (".json", ".jsonl", or ".parquet").
    input_formats
        List of acceptable input file formats (e.g., [".json", ".jsonl"]).
        Defaults to ``None`` which means all formats are accepted.
    data_class
        Class type to validate and load each data item.
        Defaults to ``None`` which means the raw dictionary is used.
    data_transform
        Function to transform the loaded data.
        If ``None``, no transformation is applied.
    name_transform
        Function to transform the name from the input file to the name
        of the output file.

    Raises
    ------
    ValueError
        If the input file path does not exist or is not a file or directory.
    """

    def save_callback(data_list: list, input_file: str, input: str):
        # The case when input is a file
        if input_file == input:
            input_rel_path = os.path.basename(input_file)
        # The case when input is a folder
        else:
            input_rel_path = os.path.relpath(input_file, input)
        if name_transform is not None:
            input_rel_path = name_transform(input_rel_path)
        save_path = str(Path(save_dir) / Path(input_rel_path).with_suffix(save_format))
        save_list_to_file(data_list, save_path, exclude_none=False)

    def get_input_files(input_path: str) -> list[str]:
        return get_files_recursively(input_path, input_formats=input_formats)

    def load_callback(input_path: str) -> list:
        return load_list_from_file(input_path, data_class)

    meta_process_data(
        inputs,
        save_callback,
        get_input_files=get_input_files,
        load_callback=load_callback,
        data_transform=data_transform,
        data_combine=None,
    )
