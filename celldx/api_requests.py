import requests
import numpy as np

from .exceptions import InvalidInputData, InvalidApiKey, NotEnoughCredits, InputArrayLengthLimitExceeded, RateLimitExceeded


def send_request_to_inference(url, api_key, array_length, data):
    rsp = requests.post(url, data=data, headers={'X-API-KEY': api_key, 'Content-Type': 'application/octet-stream'},
                        params={'array_length': array_length})
    print(array_length)
    if rsp.status_code == 200:
        if len(rsp.content) != 4096 * array_length:
            raise requests.HTTPError
        raw_array = np.frombuffer(rsp.content, dtype='float32')
        return raw_array.reshape(array_length, 1024)

    elif rsp.status_code == 400:
        raise InvalidInputData
    elif rsp.status_code == 401:
        raise InvalidApiKey
    elif rsp.status_code == 403:
        raise NotEnoughCredits
    elif rsp.status_code == 413:
        raise InputArrayLengthLimitExceeded
    elif rsp.status_code == 429:
        raise RateLimitExceeded
    else:
        raise requests.HTTPError
