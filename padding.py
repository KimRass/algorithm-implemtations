# constant_pad
# reflect_pad
# replicate_pad
# symmetric_pad

import numpy as np


def constant_pad(x, padding, constant):
    if isinstance(padding, int):
        padding = (padding, padding)

    h, w, *_ = x.shape
    padded_x = np.full(
        (h + 2*padding[0], w + 2*padding[1], *_),
        fill_value=np.clip(
            constant, np.iinfo(x.dtype).min, np.iinfo(x.dtype).max,
        ),
        dtype=x.dtype,
    )
    padded_x[padding[0]: padding[0] + h, padding[1]: padding[1] + w, ...] = x
    return padded_x


def replicate_pad(x, padding):
    if isinstance(padding, int):
        padding = (padding, padding)

    h, w, *_ = x.shape
    padded_x = np.zeros((h + 2*padding[0], w + 2*padding[1], *_), dtype=x.dtype)
    padded_x[padding[0]: padding[0] + h, padding[1]: padding[1] + w] = img
    padded_x[
        : padding[0], : padding[1], ...
    ] = img[:: -1, :: -1, ...][-padding[0]:, -padding[1]:, ...]
    padded_x[
        : padding[0], -padding[1]:, ...
    ] = img[:: -1, :: -1, ...][-padding[0]:, : padding[1], ...]
    padded_x[
        -padding[0]:, : padding[1], ...
    ] = img[:: -1, :: -1, ...][: padding[0], -padding[1]:, ...]
    padded_x[
        -padding[0]:, -padding[1]:, ...
    ] = img[:: -1, :: -1, ...][: padding[0], : padding[1], ...]
    padded_x[
        : padding[0], padding[0]: -padding[1], ...,
    ] = img[:: -1, ...][-padding[0]:, ...]
    padded_x[
        -padding[0]:, padding[0]: -padding[1], ...,
    ] = img[:: -1, ...][: padding[0], ...]
    padded_x[
        padding[0]: -padding[1], : padding[0], ...,
    ] = img[:, :: -1, ...][:, -padding[0]:, ...]
    padded_x[
        padding[0]: -padding[1], -padding[0]:, ...,
    ] = img[:, :: -1, ...][:, : padding[0], ...]
    return padded_x


if __name__ == "__main__":
    import cv2

    img_path = "./resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    padding = 100
    out1 = constant_pad(img, padding=padding, constant=-300)
    out2 = replicate_pad(img, padding=padding)

