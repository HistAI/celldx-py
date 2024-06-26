import requests
import numpy as np

from .exceptions import InvalidInputData, InvalidApiKey, NotEnoughCredits, InputArrayLengthLimitExceeded, RateLimitExceeded


def send_request_to_inference(url, api_key, array_length, data, compression):
    rsp = requests.post(url, data=data, headers={'X-API-KEY': api_key, 'Content-Type': 'application/octet-stream'},
                        params={'array_length': array_length, 'compression': compression})
    rsp_json = rsp.json()
    message = ''
    if 'detail' in rsp_json.keys():
        message = rsp_json['detail']

    if rsp.status_code == 200:
        if len(rsp.content) != 4096 * array_length:
            raise requests.HTTPError
        raw_array = np.frombuffer(rsp.content, dtype='float32')
        return raw_array.reshape(array_length, 1024)

    elif rsp.status_code == 400:
        raise InvalidInputData(message)
    elif rsp.status_code == 401:
        raise InvalidApiKey(message)
    elif rsp.status_code == 403:
        raise NotEnoughCredits(message)
    elif rsp.status_code == 413:
        raise InputArrayLengthLimitExceeded(message)
    elif rsp.status_code == 429:
        raise RateLimitExceeded(message)
    else:
        raise requests.HTTPError(message)
