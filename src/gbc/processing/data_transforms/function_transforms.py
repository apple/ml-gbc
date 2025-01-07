# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from collections.abc import Callable
from functools import reduce


def create_list_transform(transform_function: Callable) -> Callable:
    """
    Returns a function that applies a given transformation function
    to a list of objects.

    Parameters
    ----------
    transform_function
        A function to apply to each element of a list.

    Returns
    -------
    Callable
        A function that takes a list of objects and returns a new list of
        transformed objects.
    """

    def list_transform(obj_list):
        return list(map(transform_function, obj_list))

    return list_transform


def chain_transforms(*transform_functions: Callable) -> Callable:
    """
    Returns a function that chains multiple transformation functions together,
    applying them sequentially to an input.

    Parameters
    ----------
    *transform_functions : Callable
        A variable number of functions to apply sequentially.

    Returns
    -------
    Callable
        A function that takes an object and applies the chain of transformations
        to it in order.
    """

    def chained_transform(input_obj):
        return reduce(lambda obj, func: func(obj), transform_functions, input_obj)

    return chained_transform
