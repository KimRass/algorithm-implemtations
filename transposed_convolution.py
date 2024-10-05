# References:
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    # https://d2l.ai/chapter_computer-vision/transposed-conv.html
    # https://github.com/vdumoulin/conv_arithmetic

import torch


def get_output_shape(x, kernel, stride, padding, out_padding, dilation):
    """
    `padding`: The padding is applied to output. The first and last rows and
    columns will be removed from the output.
    """
    b, _, h, w = x.shape
    out_h = (h - 1)*stride[0] - 2*padding[0] + dilation[0]*(
        kernel.size(2) - 1
    ) + out_padding[0] + 1
    out_w = (w - 1)*stride[1] - 2*padding[1] + dilation[1]*(
        kernel.size(3) - 1
    ) + out_padding[1] + 1
    return b, kernel.size(1), out_h, out_w


def transposed_convolve_2d(
    x, kernel, stride=1, padding=0, out_padding=0, dilation=1,
):
    """
    `stride`: The strides are specified for intermediate results (thus
    output), not for input.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(out_padding, int):
        out_padding = (out_padding, out_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    out_shape = get_output_shape(
        x,
        kernel=kernel,
        stride=stride,
        padding=padding,
        out_padding=out_padding,
        dilation=dilation,
    )
    out = torch.zeros(size=out_shape, dtype=x.dtype, device=x.device)
    print(out.shape)
    for k in range(x.shape[1]):
        # for i in range(x.shape[2]):
        #     for j in range(x.shape[3]):
        for i in range(x.shape[2] - padding[0]):
            for j in range(x.shape[3] - padding[1]):
                a = out[
                    :,
                    :,
                    i*stride[0]: i*stride[0] + kernel.shape[2]: dilation[0],
                    j*stride[1]: j*stride[1] + kernel.shape[3]: dilation[1],
                ]
                b = torch.tensordot(x[:, k, i, j], kernel[k, :, :, :], dims=0)
                a += b
                # out[
                #     :,
                #     :,
                #     i*stride[0]: i*stride[0] + kernel.shape[2]: dilation[0],
                #     j*stride[1]: j*stride[1] + kernel.shape[3]: dilation[1],
                # ] += torch.tensordot(x[:, k, i, j], kernel[k, :, :, :], dims=0)
    return out


if __name__ == "__main__":
    import torch.nn.functional as F

    batch_size = 16
    in_channels = 4
    h = 45
    w = 24
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
    # stride = (2, 3)
    stride = (1, 1)
    # padding = (4, 3)
    padding = (1, 1)
    out_padding = (0, 0)
    dilation = 1
    out1 = F.conv_transpose2d(
        x,
        weight=kernel,
        stride=stride,
        padding=padding,
        output_padding=out_padding,
        dilation=dilation,
    )
    # print(out1.shape)
    out2 = transposed_convolve_2d(
        x, kernel=kernel, stride=stride, padding=padding, dilation=dilation,
    )
    out1[0, 0, : 5, : 5]
    out2[0, 0, : 5, : 5]
    print((out1 - out2).mean())
    # a = torch.nn.ConvTranspose2d(in_channels, out_channels, (2, 2))
    # x.shape, a.weight.shape, a(x).shape
