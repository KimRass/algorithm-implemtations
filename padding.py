# reflect_pad
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


def reflect_pad(x, padding):
    if isinstance(padding, int):
        padding = (padding, padding)

    h, w, *_ = x.shape
    padded_x = np.zeros((h + 2*padding[0], w + 2*padding[1], *_), dtype=x.dtype)
    padded_x[padding[0]: padding[0] + h, padding[1]: padding[1] + w] = x
    padded_x[
        : padding[0], : padding[1], ...,
    ] = x[:: -1, :: -1, ...][-padding[0]:, -padding[1]:, ...]
    padded_x[
        : padding[0], -padding[1]:, ...,
    ] = x[:: -1, :: -1, ...][-padding[0]:, : padding[1], ...]
    padded_x[
        -padding[0]:, : padding[1], ...,
    ] = x[:: -1, :: -1, ...][: padding[0], -padding[1]:, ...]
    padded_x[
        -padding[0]:, -padding[1]:, ...,
    ] = x[:: -1, :: -1, ...][: padding[0], : padding[1], ...]
    padded_x[
        : padding[0], padding[0]: -padding[1], ...,
    ] = x[:: -1, ...][-padding[0]:, ...]
    padded_x[
        -padding[0]:, padding[0]: -padding[1], ...,
    ] = x[:: -1, ...][: padding[0], ...]
    padded_x[
        padding[0]: -padding[1], : padding[0], ...,
    ] = x[:, :: -1, ...][:, -padding[0]:, ...]
    padded_x[
        padding[0]: -padding[1], -padding[0]:, ...,
    ] = x[:, :: -1, ...][:, : padding[0], ...]
    return padded_x


def replicate_pad(x, padding):
    if isinstance(padding, int):
        padding = (padding, padding)

    h, w, *_ = x.shape
    padded_x = np.zeros(
        (h + 2*padding[0], w + 2*padding[1], *_), dtype=x.dtype,
    )
    padded_x[padding[0]: padding[0] + h, padding[1]: padding[1] + w] = x
    padded_x[: padding[0], : padding[1], ...] = x[0, 0, ...]
    padded_x[: padding[0], -padding[1]:, ...] = x[0, -1, ...]
    padded_x[-padding[0]:, : padding[1], ...] = x[-1, 0, ...]
    padded_x[-padding[0]:, -padding[1]:, ...] = x[-1, -1, ...]
    padded_x[: padding[0], padding[0]: -padding[1], ...] = x[0, :, ...]
    padded_x[-padding[0]:, padding[0]: -padding[1], ...] = x[-1, :, ...]
    padded_x[
        padding[0]: -padding[1], : padding[0], ...,
    ] = np.repeat(x[:, 0, ...][:, None, :], repeats=padding[1], axis=1)
    padded_x[
        padding[0]: -padding[1], -padding[0]:, ...,
    ] = np.repeat(x[:, -1, ...][:, None, :], repeats=padding[1], axis=1)
    return padded_x


if __name__ == "__main__":
    import cv2
    
    from utils import show_image

    img_path = "./resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    padding = 60
    out1 = constant_pad(img, padding=padding, constant=-300)
    out2 = reflect_pad(img, padding=padding)
    out3 = replicate_pad(img, padding=padding)
    show_image(out1)
    show_image(out2)
    show_image(out3)
