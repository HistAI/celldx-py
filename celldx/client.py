import numpy as np

from typing import List, Union
from .validations import bool_validate_all_elements_same_type
from .utils import read_files_and_resize_cv2, resize_arrays, convert_ndarrays_list_to_ndarray
from .api_requests import send_request_to_inference


API_URL = 'https://api.celldx.net'
MODEL_ENDPOINT = '/inference/run'


class HibouApiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_data(self, data: Union[str, List[str], np.ndarray, List[np.ndarray]]):
        if isinstance(data, str):
            data = [data]
        if isinstance(data, list) and len(data) > 0 and bool_validate_all_elements_same_type(data, str):
            outputs = read_files_and_resize_cv2(data)
        elif isinstance(data, list) and len(data) > 0 and bool_validate_all_elements_same_type(data, np.ndarray):
            outputs = resize_arrays(data)
        elif isinstance(data, np.ndarray) and len(data.shape) == 3:
            outputs = resize_arrays([data])
        elif isinstance(data, np.ndarray) and len(data.shape) == 4:
            outputs = resize_arrays(data)
        else:
            raise TypeError("Invalid data type")
        outputs = convert_ndarrays_list_to_ndarray(outputs)

        response = send_request_to_inference(API_URL+MODEL_ENDPOINT, self.api_key, outputs.shape[0], outputs.tobytes())
        return response
