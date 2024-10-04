# References:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

import torch


def get_output_shape(x, kernel, padding, stride, dilation):
    b, _, h, w = x.shape
    return (
        b,
        kernel.size(0),
        int(
            (h + 2 * padding[0] - dilation[0] * (kernel.size(2) - 1) - 1) / stride[0] + 1
        ),
        int(
            (w + 2 * padding[1] - dilation[1] * (kernel.size(3) - 1) - 1) / stride[1] + 1
        ),
    )


def pad(x):
    b, c, h, w = x.shape
    padded = torch.zeros(
        size=(b, c, h + padding[0] * 2, w + padding[1] * 2),
        dtype=x.dtype,
        device=x.device,
    )
    padded[:, :, padding[0]: padding[0] + h, padding[1]: padding[1] + w] = x
    return padded


def convolve_2d(x, kernel, padding=0, stride=1, dilation=1):
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    out_shape = get_output_shape(
        x, kernel=kernel, padding=padding, stride=stride, dilation=dilation,
    )
    out = torch.zeros(size=out_shape, dtype=x.dtype, device=x.device)
    x = pad(x)
    for k in range(kernel.size(0)):
        for i in range(out_shape[2]):
            for j in range(out_shape[3]):
                x_region = x[
                    :,
                    :,
                    i * stride[0]: i * stride[0] + kernel.size(2),
                    j * stride[1]: j * stride[1] + kernel.size(3),
                ]
                kernel_region = kernel[k, ...].repeat(out_shape[0], 1, 1, 1)
                conv_out = torch.sum(
                    x_region * kernel_region, dim=(1, 2, 3),
                )
                out[:, k, i, j] = conv_out
    return out


if __name__ == "__main__":
    import torch.nn.functional as F

    batch_size = 16
    in_channels = 4
    h = 7
    w = 8
    x = torch.randn(
        size=(batch_size, in_channels, h, w), dtype=torch.float32,
    )
    out_channels = 5
    kernel = torch.randn(
        size=(out_channels, in_channels, 3, 5), dtype=torch.float32,
    )
    stride = 3
    padding = (2, 4)
    dilation = 1
    out1 = F.conv2d(
        x, weight=kernel, stride=stride, padding=padding, dilation=dilation,
    )
    out2 = convolve_2d(
        x, kernel=kernel, stride=stride, padding=padding, dilation=dilation,
    )
    print((out1 - out2).mean())
