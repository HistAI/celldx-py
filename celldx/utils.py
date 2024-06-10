import cv2
from .exceptions import FileReadError
import numpy as np
from .validations import validate_paths, validate_ndarray_dtype_uint8, validate_array_shapes_resizable, \
    validate_array_shape_resizable, validate_exact_array_shape, bool_validate_exact_array_shape


def read_file_cv2(path: str):
    output = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    if output is None:
        raise FileReadError("File read error")
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


def read_files_and_resize_cv2(paths):
    validate_paths(paths)
    outputs = list()
    for path in paths:
        output = read_file_cv2(path)
        validate_array_shape_resizable(output)
        output = resize_array(output)
        outputs.append(output)
    return outputs


def resize_array(array):
    if bool_validate_exact_array_shape(array):
        return array
    return cv2.resize(array, (224, 224), interpolation=cv2.INTER_LINEAR)


def resize_arrays(arrays):
    validate_array_shapes_resizable(arrays)
    outputs = list()
    for array in arrays:
        validate_ndarray_dtype_uint8(array)
        outputs.append(resize_array(array))
    return outputs


def convert_ndarrays_list_to_ndarray(arrays):
    for ndarr in arrays:
        validate_exact_array_shape(ndarr)
    return np.stack(arrays, 0)




