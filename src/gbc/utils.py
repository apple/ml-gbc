# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import os
import sys
import copy
import json
import time
import logging
import importlib
import pandas as pd

from typing import Any, Callable, Optional, Type
from pathlib import Path
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


def get_gbc_logger() -> logging.Logger:
    """
    Get the logger for GBC captioning.

    Returns
    -------
    logging.Logger
        The logger instance for GBC captioning.
    """
    return logging.getLogger("GBC logging")


def setup_gbc_logger(level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up the GBC logger with a specified logging level.

    Parameters
    ----------
    level
        The logging level, by default logging.DEBUG.
    """
    logger = get_gbc_logger()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_list_from_file(
    load_file: str, class_type: Optional[Type] = None, disable_tqdm: bool = False
) -> list:
    """
    Load a list of data from a file, with support for JSON, JSONL, and Parquet.

    When the class type is provided, the data will be converted to
    the specified class type using the ``model_validate`` method.

    Parameters
    ----------
    load_file
        The path to the file to load the data from.
    class_type
        The class type to validate each data item, by default None.
        When provided, the class must implement the ``model_validate`` method.

    Returns
    -------
    list
        The list of loaded data items, converted to the specified class type if any.

    Raises
    ------
    AssertionError
        If the file extension is not supported.
    """
    logger = get_gbc_logger()
    logger.info(f"Loading from {load_file}...")
    _, ext = os.path.splitext(load_file)
    assert ext in [".json", ".jsonl", ".parquet"], f"Unsupported file type: {ext}"

    with open(load_file, "r") as f:
        if ext == ".json":
            data_list = json.load(f)
        elif ext == ".jsonl":
            data_list = [json.loads(line) for line in f]
        elif ext == ".parquet":
            data_list = pd.read_parquet(load_file).to_dict(orient="records")

    if class_type is not None:
        disable = len(data_list) < 50000 or disable_tqdm
        data_list = [
            class_type.model_validate(data) for data in tqdm(data_list, disable=disable)
        ]
    return data_list


def save_list_to_file(data_list: list, save_file: str, **kwargs):
    """
    Save a list of data to a file, with support for JSON, JSONL, and Parquet.

    If the content of the data list is not a dictionary, it will be
    converted to a dictionary using the ``model_dump`` method.

    Parameters
    ----------
    data_list : list
        The list of data items to save. Items can be either dictionaries
        or models that implement the ``model_dump`` method
        (e.g. `Pydantic <https://docs.pydantic.dev/latest/>`_ models).
    save_file : str
        The path to the file to save the data to.
    **kwargs
        Additional keyword arguments to pass to the ``model_dump`` method.

    Raises
    ------
    AssertionError
        If the file extension is not supported.
    """

    def dump_data(obj):
        if isinstance(obj, dict):
            return obj
        return obj.model_dump(**kwargs)

    logger = get_gbc_logger()
    logger.info(f"Saving to {save_file}...")
    _, ext = os.path.splitext(save_file)
    assert ext in [".json", ".jsonl", ".parquet"], f"Unsupported file type: {ext}"

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    disable = len(data_list) < 50000

    with open(save_file, "w") as f:
        if ext == ".json":
            data_dicts = [dump_data(data) for data in tqdm(data_list, disable=disable)]
            json.dump(data_dicts, f, indent=4)
        elif ext == ".jsonl":
            for data in tqdm(data_list, disable=disable):
                f.write(json.dumps(dump_data(data)) + "\n")
        elif ext == ".parquet":
            data_dicts = [dump_data(data) for data in tqdm(data_list, disable=disable)]
            pd.DataFrame(data_dicts).to_parquet(save_file, index=False)


def instantiate_class(module_name, class_name):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_files_recursively(
    file_path: str, input_formats: Optional[list[str]] = None
) -> list[str]:
    """
    Recursively get files with specified formats from a directory or single file.

    Parameters
    ----------
    file_path
        The path to the directory or single file.
    input_formats
        List of file formats to include (e.g., ['.jpg', '.png']).

    Returns
    -------
    list[str]
        List of file paths matching the specified formats.

    Raises
    ------
    ValueError
        If the provided path does not exist.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist.")

    if input_formats is None:
        allowed_patterns = ["*"]
    else:
        allowed_patterns = [f"*{fmt}" for fmt in input_formats]

    if os.path.isdir(file_path):
        return [
            str(path)
            for pattern in allowed_patterns
            for path in Path(file_path).rglob(pattern)
            if path.is_file()
        ]
    elif os.path.isfile(file_path) and any(
        file_path.endswith(fmt) for fmt in input_formats
    ):
        return [file_path]
    else:
        return []


def get_images_recursively(folder_path: str) -> list[str]:
    """
    Get all images recursively from a folder.

    Parameters
    ----------
    folder_path
        The path to the folder.

    Returns
    -------
    list[str]
        The list of paths to images found recursively in the folder.

    Raises
    ------
    ValueError
        If the provided path does not exist.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The path {folder_path} does not exist.")

    allowed_patterns = [
        "*.[Pp][Nn][Gg]",
        "*.[Jj][Pp][Gg]",
        "*.[Jj][Pp][Ee][Gg]",
        "*.[Ww][Ee][Bb][Pp]",
        "*.[Gg][Ii][Ff]",
    ]

    image_path_list = [
        str(path)
        for pattern in allowed_patterns
        for path in Path(folder_path).rglob(pattern)
    ]

    return image_path_list


def backoff(func: Callable, retry_count: int = 1, max_retries: int = 3):

    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            remaining_retries = max_retries - retry_count
            if remaining_retries < 1:
                print(f"{max_retries} retries failed, bailing....")
                raise exc

            sleep_time = 5**retry_count
            print(
                f"Caught exception: {exc}, sleeping {sleep_time} sec and retrying"
                f" {remaining_retries} times."
            )
            time.sleep(sleep_time)
            return backoff(func, retry_count=retry_count + 1, max_retries=max_retries)(
                *args, **kwargs
            )

    return inner


class ImageCache(BaseModel):
    """
    Handles image loading and caching.

    This class avoids loading the same image multiple times when
    it is used by different actions by caching the loaded image.

    Attributes
    ----------
    img_path : str | None
        Path to the image file (relative or absolute).
    img_url : str | None
        URL to fetch the image if the local file is unavailable or invalid.
    image : Image.Image | None
        Cached image data in RGB format.
        This is not expected to be provided by the user.
    """

    img_path: Optional[str] = None
    img_url: Optional[str] = None
    image: Any = None

    def get_image(self) -> Optional[Image.Image]:
        """
        Loads and returns the image as PIL Image.
        If the image is already cached, the cached version is returned.
        Logs warnings for missing or invalid files and
        attempts to fetch from the URL if the local file fails.

        Returns
        -------
        Image.Image or None
            The loaded image in RGB format, or `None` if the image cannot be loaded.
        """
        logger = get_gbc_logger()

        # Return cached image if already loaded
        if self.image is not None:
            return self.image

        # Attempt to load the local image
        if self.img_path:
            if os.path.exists(self.img_path):
                try:
                    self.image = Image.open(self.img_path).convert("RGB")
                    return self.image
                except (UnidentifiedImageError, IOError):
                    logger.warning(
                        f"File exists but is not a valid image: {self.img_path}"
                    )
            else:
                logger.warning(f"Image file does not exist: {self.img_path}")

        # Attempt to load the image from the URL
        if self.img_url:
            try:
                import requests

                response = requests.get(self.img_url, stream=True, timeout=10)
                response.raise_for_status()  # Raise HTTPError for bad responses
                self.image = Image.open(response.raw).convert("RGB")
                return self.image
            except (requests.RequestException, UnidentifiedImageError, IOError) as e:
                logger.warning(
                    "Failed to fetch or validate image from URL: "
                    f"{self.img_url} | Error: {e}"
                )

        return None
