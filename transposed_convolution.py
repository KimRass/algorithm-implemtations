# References:
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    # https://d2l.ai/chapter_computer-vision/transposed-conv.html
    # https://github.com/vdumoulin/conv_arithmetic

import torch


def get_output_shape(x, kernel, stride, padding, out_padding, dilation):
    b, _, h, w = x.shape
    return (
        b,
        kernel.size(1),
        (h - 1) * stride[0] - 2 * padding[0] + dilation * (
            kernel.size(2) - 1
        ) + out_padding[0] + 1,
        (w - 1) * stride[1] - 2 * padding[1] + dilation * (
            kernel.size(3) - 1
        ) + out_padding[1] + 1,
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


def transposed_convolve_2d(
    x, kernel, stride=1, padding=0, out_padding=0, dilation=1,
):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(out_padding, int):
        out_padding = (out_padding, out_padding)

    out_shape = get_output_shape(
        x,
        kernel=kernel,
        stride=stride,
        padding=padding,
        out_padding=out_padding,
        dilation=dilation,
    )
    out = torch.zeros(size=out_shape, dtype=x.dtype, device=x.device)
    x = pad(x)
    # print(x.shape, kernel.shape, out_shape)
    for k in range(x.shape[1]):
        for i in range(x.shape[2]):
            for j in range(x.shape[3]):
                a = out[
                    ..., i: i + kernel.shape[2], j: j + kernel.shape[3]
                ]
                b = torch.tensordot(x[:, k, i, j], kernel[k, ...], dims=0)
                a += b
    return out


if __name__ == "__main__":
    import torch.nn.functional as F

    batch_size = 16
    in_channels = 4
    h = 2
    w = 2
    x = torch.randn(
        size=(batch_size, in_channels, h, w), dtype=torch.float32,
    )
    out_channels = 5
    kernel_h = 2
    kernel_w = 2
    kernel = torch.randn(
        size=(in_channels, out_channels, kernel_h, kernel_w),
        dtype=torch.float32,
    )
    stride = (1, 1)
    padding = (0, 0)
    out_padding = (0, 0)
    dilation = 1
    get_output_shape(
        x,
        kernel=kernel,
        stride=stride,
        padding=padding,
        out_padding=out_padding,
        dilation=dilation,
    )
    out1 = F.conv_transpose2d(
        x,
        weight=kernel,
        stride=stride,
        padding=padding,
        output_padding=out_padding,
        dilation=dilation,
    )
    out1.shape
    out2 = transposed_convolve_2d(
        x, kernel=kernel, stride=stride, padding=padding, dilation=dilation,
    )
    print((out1 - out2).mean())
    # a = torch.nn.ConvTranspose2d(in_channels, out_channels, (2, 2))
    # x.shape, a.weight.shape, a(x).shape
