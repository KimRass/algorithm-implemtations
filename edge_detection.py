# References:
    # https://en.wikipedia.org/wiki/Sobel_operator
    # https://gaussian37.github.io/vision-concept-edge_detection/

import numpy as np

from convolution_numpy import convolve
from blurring import gaussian_blur


def sobel_edge_detect(x, to_uint8=True):
    x_kernel = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32 # `np.finfo(np.float32).max` > `((255*4)**2)*2`
    )
    y_kernel = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32
    )
    x_grad = convolve(
        x,
        kernel=x_kernel,
        padding=(x_kernel.shape[0]//2, x_kernel.shape[1]//2),
        padding_mode="replicate",
    )
    y_grad = convolve(
        x,
        kernel=y_kernel,
        padding=(y_kernel.shape[0]//2, y_kernel.shape[1]//2),
        padding_mode="replicate",
    )
    if x.ndim == 3:
        x_grad = np.mean(x_grad, axis=2)
        y_grad = np.mean(y_grad, axis=2)

    grad_mag = (x_grad**2 + y_grad**2)**0.5
    if to_uint8:
        grad_mag /= (((255*4)**2)*2)**0.5
        grad_mag *= 255
        grad_mag = grad_mag.astype(np.uint8)
    grad_dir = np.arctan2(y_grad, x_grad)
    return grad_mag, grad_dir


def non_max_suppress(grad_mag, grad_dir):
    # Initialize an empty suppressed image
    suppressed = np.zeros_like(grad_mag)
    angle = grad_dir * 180 / np.pi
    angle[angle < 0] += 180  # Normalize direction to [0, 180]

    rows, cols = grad_mag.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            try:
                # Determine the neighboring pixels to compare
                q, r = 255, 255

                # Check direction and compare neighbors in that direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = grad_mag[i, j + 1]
                    r = grad_mag[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = grad_mag[i + 1, j - 1]
                    r = grad_mag[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = grad_mag[i + 1, j]
                    r = grad_mag[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = grad_mag[i - 1, j - 1]
                    r = grad_mag[i + 1, j + 1]

                # Suppress pixel if it is not the local maximum
                if grad_mag[i, j] >= q and grad_mag[i, j] >= r:
                    suppressed[i, j] = grad_mag[i, j]
                else:
                    suppressed[i, j] = 0
            except IndexError:
                pass
    return suppressed


def canny_edge_detect(x, kernel_size, std):
    x = gaussian_blur(x, kernel_size=kernel_size, std=std)
    grad_mag, grad_dir = sobel_edge_detect(x, to_uint8=False)
    out = non_max_suppress(grad_mag=grad_mag, grad_dir=grad_dir)


if __name__ == "__main__":
    import cv2

    from utils import show_image

    img_path = "/Users/jongbeomkim/Desktop/workspace/image-processing-algorithms/resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    sobel_out, _ = sobel_edge_detect(img)
    show_image(sobel_out)
