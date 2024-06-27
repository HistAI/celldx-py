# celldx-py

Use the Python Library to utilize the inference API from your code. `celldx-py` package allows to interact with the CellDX model inference API from your python code in a convenient way.

It provides a set of functions to help to preprocess and validate an input/output data.

## Getting started

Make sure that the following requirements are met before starting work.

### 1. Install Python 3.8 or above

If you don't have Python installed yet, you can get the most recent version [here](https://www.python.org/downloads/).

### 2. Install celldx-py package

To install CellDX python library, open terminal or the command prompt, and enter the following command:

```
pip install celldx-py
```

If it is already installed, make sure you are using an up-to-date version by entering the following command:

```
pip install celldx-py --upgrade
```

### 3. Generate an API key

If you don't have an API key yet, generate it on the CellDX [portal](https://celldx.hist.ai/workplace/settings).

## Authentication

To interact with the model inference API, you need to authenticate by instantiating a client.
It is created directly from an API key.

```
from celldx import HibouApiClient

API_KEY = "YOUR_API_KEY"
hibou_client = HibouApiClient(API_KEY)
```

## Inference

Send your data to the model inference and get an output. You can use the instantiated client to send your data to be processed.

Here is a simple example of getting your input data and passing it to the required function.

```
import cv2

...

data = cv2.imread('your_image.jpg')

features = hibou_client.process_data(data=data, resize=True, compression=False)
```

`process_data` function processes the input data and sends a request to the inference API.

The corresponding model inference api endpoint accepts a binary form of `numpy.ndarray` with strictly defined shape (N, 224, 224, 3), where 'N' has a limit of 512. The data-type of the arrays elements: uint8. The input data can also be an encoded binary representation of the compressed array.

`process_data` function takes the following parameters:

`data` (Union[str, List[str], numpy.ndarray, List[numpy.ndarray]]) - the input data to be processed. It can be a single string file path, a list of string file paths, a single numpy array, or a list of numpy arrays. The parameter can take `numpy.ndarray` with a length of shape equal to 3 or 4 (`(224, 224, 3)` or `(N, 224, 224, 3)` or suitable shapes which require resizing). If the parameter is a list of numpy arrays, each element must have a length of shape equal to 3 (`(224, 224, 3)` or suitable shapes which require resizing). The data-type of the arrays elements: uint8.

`resize` (bool) parameter indicates the need to transform data into the required shape. The default value of the `resize` parameter is `False`. The required shape of a single image array: (224, 224, 3).

`compression` (bool) parameter indicates the need to compress the input data. The default value is `False`.
If `compression` parameter is `True`, then the arrays are JPEG compressed. This may reduce the request body size, but affects the inference results.

Below are some examples of using the function.

The `data` parameter can be an image path:

```
features = hibou_client.process_data(data='/home/user/my_image.jpg')
```

Or a list of image paths:

```
paths_list = ['/home/user/my_image_1.jpg', '/home/user/my_image_2.jpg']
features = hibou_client.process_data(data=paths_list, resize=True)
```

The function returns a numpy.ndarray with the shape of `(N, 1024)`, where N - the length of input data array. The data-type of the arrays elements: float32.

It is assumed that the data is transmitted in RGB color space.
