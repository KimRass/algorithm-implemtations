# References:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    # https://github.com/vdumoulin/conv_arithmetic

import torch


def get_output_shape(x, kernel, stride, padding, dilation):
    b, _, h, w = x.shape
    out_h = int(
        (
            h + 2*padding[0] - dilation[0]*(kernel.size(2) - 1) - 1
        ) / stride[0] + 1
    )
    out_w = int(
        (
            w + 2*padding[1] - dilation[1]*(kernel.size(3) - 1) - 1
        ) / stride[1] + 1
    )
    return b, kernel.size(0), out_h, out_w


def pad(x, padding):
    b, c, h, w = x.shape
    padded = torch.zeros(
        size=(b, c, h + 2*padding[0], w + 2*padding[1]),
        dtype=x.dtype,
        device=x.device,
    )
    padded[:, :, padding[0]: padding[0] + h, padding[1]: padding[1] + w] = x
    return padded


def convolve_2d(x, kernel, stride=1, padding=0, dilation=1):
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
    x = pad(x, padding=padding)
    new_kernel_h = kernel.size(2) + (kernel.size(2) - 1)*(dilation[0] - 1)
    new_kernel_w = kernel.size(3) + (kernel.size(3) - 1)*(dilation[1] - 1)
    for k in range(out_shape[1]):
        for i in range(out_shape[2]):
            for j in range(out_shape[3]):
                x_region = x[
                    :,
                    :,
                    i*stride[0]: i*stride[0] + new_kernel_h: dilation[0],
                    j*stride[1]: j*stride[1] + new_kernel_w: dilation[1],
                ]
                kernel_region = kernel[k, :, :, :].repeat(out_shape[0], 1, 1, 1)
                conv_out = torch.sum(
                    x_region*kernel_region, dim=(1, 2, 3),
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
    kernel_h = 3
    kernel_w = 5
    kernel = torch.randn(
        size=(out_channels, in_channels, kernel_h, kernel_w),
        dtype=torch.float32,
    )
    stride = (2, 3)
    padding = (2, 4)
    dilation = 3
    out1 = F.conv2d(
        x, weight=kernel, stride=stride, padding=padding, dilation=dilation,
    )
    out2 = convolve_2d(
        x, kernel=kernel, stride=stride, padding=padding, dilation=dilation,
    )
    print(torch.abs(out1 - out2).mean())
