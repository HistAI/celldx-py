import numpy as np

from typing import List, Union
from .validations import bool_validate_all_elements_same_type, validate_exact_array_shapes
from .utils import read_files_and_resize_cv2, convert_ndarrays_list_to_ndarray, validate_or_resize_array, \
    compress_and_convert_array_to_bytes
from .api_requests import send_request_to_inference


API_URL = 'https://api.celldx.net'
MODEL_ENDPOINT = '/inference/run'


class HibouApiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_data(self, data: Union[str, List[str], np.ndarray, List[np.ndarray]], resize: bool = False, compression: bool = False):
        if isinstance(data, str):
            data = [data]
        if isinstance(data, list) and len(data) > 0 and bool_validate_all_elements_same_type(data, str):
            outputs = read_files_and_resize_cv2(data, resize)
        elif isinstance(data, list) and len(data) > 0 and bool_validate_all_elements_same_type(data, np.ndarray):
            outputs = validate_or_resize_array(data, resize)
        elif isinstance(data, np.ndarray) and len(data.shape) == 3:
            outputs = validate_or_resize_array([data], resize)
        elif isinstance(data, np.ndarray) and len(data.shape) == 4:
            outputs = validate_or_resize_array(data, resize)
        else:
            raise TypeError("Invalid data type")
        array_length = len(outputs)
        if compression:
            outputs = compress_and_convert_array_to_bytes(outputs)
        else:
            outputs = convert_ndarrays_list_to_ndarray(outputs)
            outputs = outputs.tobytes()
        response = send_request_to_inference(API_URL+MODEL_ENDPOINT, self.api_key, array_length, outputs, compression)
        return response
