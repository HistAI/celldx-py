import cv2
from .exceptions import FileReadError
import numpy as np
from .validations import validate_paths, validate_ndarray_dtype_uint8, validate_array_shapes_resizable, \
    validate_array_shape_resizable, validate_exact_array_shape, bool_validate_exact_array_shape, \
    validate_exact_array_shapes


def read_file_cv2(path: str):
    output = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    if output is None:
        raise FileReadError("File read error")
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


def read_files_and_resize_cv2(paths, resize):
    validate_paths(paths)
    outputs = list()
    for path in paths:
        output = read_file_cv2(path)
        if resize:
            validate_array_shape_resizable(output)
            output = resize_array(output)
        else:
            validate_exact_array_shape(output)
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


def validate_or_resize_array(arrays, resize):
    if resize:
        return resize_arrays(arrays)
    else:
        validate_exact_array_shapes(arrays)
        return arrays


def compress_and_convert_array_to_bytes(arrays):
    compressed_arrays = list()
    for array in arrays:
        _, compressed_array = cv2.imencode('.jpg', array, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        compressed_arrays.append(compressed_array)
    return compressed_arrays_to_bytes(compressed_arrays)


def compressed_arrays_to_bytes(compressed_arrays):
    compressed_arrays_bytes = []
    for array in compressed_arrays:
        array_bytes = array.tobytes()
        length = len(array_bytes)
        compressed_arrays_bytes.append(length.to_bytes(4, byteorder='big') + array_bytes)
    compressed_arrays_bytes = b''.join(compressed_arrays_bytes)
    return compressed_arrays_bytes


def convert_ndarrays_list_to_ndarray(arrays):
    for ndarr in arrays:
        validate_exact_array_shape(ndarr)
    return np.stack(arrays, 0)




