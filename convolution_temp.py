# References:
    # https://github.com/detkov/Convolution-From-Scratch/blob/main/convolution.py
    # https://towardsdatascience.com/tensorflow-for-computer-vision-how-to-implement-convolutions-from-scratch-in-python-609158c24f82
    # https://d2l.ai/chapter_computer-vision/transposed-conv.html
# References:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
#         self.stride = (stride, stride) if isinstance(stride, int) else stride
#         self.padding = (padding, padding) if isinstance(padding, int) else padding

#         self.weight = self._get_kernel()

#     def _get_output_shape(self, input):
#         b, _, h, w = input.shape
#         return (
#             b,
#             self.out_channels,
#             math.floor((h + self.padding[0] * 2 - (self.kernel.size(2) - 1) - 1) // self.stride[0] + 1),
#             math.floor((w + self.padding[1] * 2 - (self.kernel.size(3) - 1) - 1) // self.stride[1] + 1),
#         )
    
#     def _pad(self, input):
#         b, c, h, w = input.shape
#         padded = torch.zeros(
#             size=(b, c, h + self.padding[0] * 2, w + self.padding[1] * 2),
#             dtype=input.dtype,
#             device=input.device,
#         )
#         padded[:, :, self.padding[0]: self.padding[0] + h, self.padding[1]: self.padding[1] + w] = input
#         return padded

#     def _get_kernel(self):
#         kernel = torch.randn(
#             size=(self.out_channels, self.in_channels, self.kernel.size(2), self.kernel.size(3)),
#             requires_grad=True,
#         )
#         return kernel

#     def forward(self, input):
#         out_shape = self._get_output_shape(input)
#         input = self._pad(input)

#         out = torch.zeros(size=out_shape, dtype=input.dtype, device=input.device)
#         # Initialize
#         for k in range(out_shape[1]):
#             for i in range(0, out_shape[2], self.stride[0]):
#                 for j in range(0, out_shape[3], self.stride[0]):
#                     out[:, k, i, j] = torch.sum(
#                         input[
#                             :,
#                             :,
#                             i: i + self.kernel.size(2),
#                             j: j + kernel.size(3)
#                         ] * self.weight[k: k + 1, ...].repeat(out_shape[0], 1, 1, 1)
#                     )
#         return out


# class ConvTransposed2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()


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
    # torch.equal(out1, out2)





# def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#     super().__init__()

#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
#     self.stride = (stride, stride) if isinstance(stride, int) else stride
#     self.padding = (padding, padding) if isinstance(padding, int) else padding

#     self.weight = self._get_kernel()



# def _get_kernel(self):
#     kernel = torch.randn(
#         size=(self.out_channels, self.in_channels, self.kernel.size(2), self.kernel.size(3)),
#         requires_grad=True,
#     )
#     return kernel



# if __name__ == "__main__":
#     batch_size = 1
#     in_channels = 4
#     input = torch.randn(
#         size=(batch_size, in_channels, 3, 5), dtype=torch.float32,
#     )
#     # input.unfold(dimension=2, size=3, step=2).shape

#     kernel_size = (3, 5)

#     conv1 = nn.Conv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#     )
#     conv1.weight.shape
#     out1 = conv1(input)
#     conv2 = Conv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#     )
#     out2 = conv2(input)
#     out1[0, 0, 0, ...]
#     out2[0, 0, 0, ...]
#     out1.shape
#     out2
#     torch.equal(out1, out2)
    
    
#     conv1.weight.shape, conv2.weight.shape



#     # x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])[None, None, ...]
#     # k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])[None, None, ...]
#     # F.conv_transpose2d(input=x, weight=k)