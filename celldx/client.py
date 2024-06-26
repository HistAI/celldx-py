import numpy as np

from typing import List, Union
from .validations import bool_validate_all_elements_same_type, validate_length_limit
from .utils import (
    read_files_and_resize_cv2,
    convert_ndarrays_list_to_ndarray,
    validate_or_resize_array,
    compress_and_convert_array_to_bytes,
)
from .api_requests import send_request_to_inference


API_URL = "https://api.celldx.net"
MODEL_ENDPOINT = "/inference/run"


class HibouApiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_data(
        self,
        data: Union[str, List[str], np.ndarray, List[np.ndarray]],
        resize: bool = False,
        compression: bool = False,
    ):
        """
        Process the input data and send a request to the inference API.

        Args:
            data (Union[str, List[str], np.ndarray, List[np.ndarray]]): The input data to be processed. It can be a single string file path, a list of string file paths, a single numpy array, or a list of numpy arrays.

            resize (bool): The parameter to specify whether to scale the input data to the required resolution/shape or not.

            compression (bool): The parameter to specify whether to compress the input data before sending or not. JPEG compression is used. This can help reduce request transmission time, but affects the inference results.

        Returns:
            The response from the inference API - features, the ndarray of shape (N, 1024).

        Raises:
            TypeError: If the input data type is invalid.
            FileReadError: The file at the provided path can't be read.
            InvalidInputData: Invalid input data provided.
            InvalidApiKey: Invalid API key provided.
            NotEnoughCredits: Not enough credits for the request.
            InputArrayLengthLimitExceeded: The input data array length limit is exceeded.
            RateLimitExceeded: The request rate limit has been exceeded.
            FileNotFoundError: The file at the provided path is not found.
        """
        if isinstance(data, str):
            data = [data]
        validate_length_limit(data)
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
        response = send_request_to_inference(API_URL + MODEL_ENDPOINT, self.api_key, array_length, outputs, compression)
        return response
