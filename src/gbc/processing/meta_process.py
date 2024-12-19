# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from typing import Callable, Optional, TypeVar

from gbc.utils import load_list_from_file, get_gbc_logger


InputType = TypeVar("InputType")
LoadedDataType = TypeVar("LoadedDataType")
TransformedDataType = TypeVar("TransformedDataType")


def meta_process_data(
    inputs: list[InputType],
    save_callback: Callable[
        [TransformedDataType, Optional[str], Optional[InputType]], None
    ],
    *,
    get_input_files: Optional[Callable[[InputType], list[str]]] = None,
    load_callback: Optional[Callable[[str], LoadedDataType]] = None,
    data_transform: Optional[Callable[[LoadedDataType], TransformedDataType]] = None,
    data_combine: Optional[
        Callable[[TransformedDataType, TransformedDataType], TransformedDataType]
    ] = None,
) -> None:
    """
    Processes and combines data from multiple input sources.

    This function provides a flexible way to handle data processing by allowing
    the user to specify custom functions for loading, transforming, combining,
    and saving data. The processing workflow is as follows:

    1. Extract file paths from each input item using ``get_input_files``.
    2. Load data from each file using ``load_callback``.
    3. Optionally transform the loaded data using ``data_transform``.
    4. Optionally combine transformed data using ``data_combine``.
    5. Save the final result using ``save_callback``.

    Parameters
    ----------
    inputs
        List of input items to be processed.
    save_callback
        Function to save the processed data. If ``data_combine`` is not provided,
        ``save_callback`` is called for each data file.
        Otherwise, ``save_callback`` is called for the combined result.
        It takes input file path and input as arguments.
    get_input_files
        Function to extract file paths from each input item.
        If ``None``, the input items are used directly as the file paths.
    load_callback
        Function to load data from a file path.
        If ``None``, :func:`~gbc.utils.load_list_from_file` is used.
    data_transform
        Function to transform the loaded data.
        If ``None``, no transformation is applied.
    data_combine
        Function to combine transformed data from two files
        If ``None``, the data is saved individually using ``save_callback``.
    """
    logger = get_gbc_logger()
    combined_result = None
    all_input_files = []
    all_inputs = []
    for input in inputs:
        if get_input_files is not None:
            input_files = get_input_files(input)
        else:
            input_files = [input]
        for input_file in input_files:
            if load_callback is not None:
                data = load_callback(input_file)
            else:
                data = load_list_from_file(input_file)
            if data_transform is not None:
                logger.info(f"Processing {input_file}...")
                data = data_transform(data)
            if data_combine is not None:
                if combined_result is None:
                    combined_result = data
                else:
                    combined_result = data_combine(combined_result, data)
                all_input_files.append(input_file)
                all_inputs.append(data)
            else:
                save_callback(data, input_file=input_file, input=input)
    if combined_result is not None and save_callback is not None:
        save_callback(combined_result, input_file=all_input_files, input=all_inputs)
