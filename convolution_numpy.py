import numpy as np

from padding import reflect_pad, replicate_pad


def get_output_shape(x, kernel, stride, padding):
    h, w, *_ = x.shape
    out_h = int(
        (h + 2*padding[0] - (kernel.shape[0] - 1) - 1) / stride[0] + 1
    )
    out_w = int(
        (w + 2*padding[1] - (kernel.shape[1] - 1) - 1) / stride[1] + 1
    )
    return out_h, out_w, *_


def convolve(x, kernel, stride=1, padding=0, padding_mode="replicate"):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    out_shape = get_output_shape(
        x, kernel=kernel, stride=stride, padding=padding,
    )
    if x.ndim == 3:
        kernel = kernel[..., None].repeat(repeats=out_shape[2], axis=2)
    out = np.zeros(out_shape, dtype=np.float32)
    if padding_mode == "replicate":
        padded_x = replicate_pad(x, padding=padding)
    elif padding_mode == "reflect":
        padded_x = reflect_pad(x, padding=padding)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            x_region = padded_x[
                i*stride[0]: i*stride[0] + kernel.shape[0],
                j*stride[1]: j*stride[1] + kernel.shape[1],
                ...,
            ]
            conv_out = np.sum(x_region*kernel, axis=(0, 1))
            out[i, j, ...] = conv_out
    return out
