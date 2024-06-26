import os
import numpy as np

from .exceptions import InputArrayLengthLimitExceeded


ARRAY_LENGTH_LIMIT = 512


def validate_length_limit(data):
    if isinstance(data, list) or isinstance(data, np.ndarray):
        if len(data) > ARRAY_LENGTH_LIMIT:
            raise InputArrayLengthLimitExceeded("Input array length limit exceeded")
    else:
        raise TypeError("Invalid data type")


def validate_paths(paths):
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError


def bool_validate_all_elements_same_type(elements, element_type):
    for element in elements:
        if not isinstance(element, element_type):
            return False
    return True


def validate_array_shapes_resizable(arrays):
    for array in arrays:
        validate_array_shape_resizable(array)


def validate_array_shape_resizable(array):
    if len(array.shape) != 3 or array.shape[2] != 3:
        raise TypeError("Invalid array shape")


def validate_exact_array_shape(array):
    if (
        len(array.shape) != 3
        or array.shape[0] != 224
        or array.shape[1] != 224
        or array.shape[2] != 3
    ):
        raise TypeError("Invalid array shape")


def validate_exact_array_shapes(arrays):
    for array in arrays:
        validate_exact_array_shape(array)


def bool_validate_exact_array_shape(array):
    if (
        len(array.shape) != 3
        or array.shape[0] != 224
        or array.shape[1] != 224
        or array.shape[2] != 3
    ):
        return False
    return True


def validate_ndarray_dtype_uint8(array):
    if array.dtype != np.dtype("uint8"):
        raise TypeError("Invalid array dtype")
