import numpy as np

from convolution_numpy import convolve


def get_gaussian_kernel(kernel_size, std):
    assert kernel_size % 2 == 1
    vals = np.arange(
        -kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=np.float64,
    )
    xs, ys = np.meshgrid(vals, vals)
    kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * std ** 2))
    kernel /= np.sum(kernel)
    return kernel


def gaussian_blur(img, kernel_size, std):
    kernel = get_gaussian_kernel(kernel_size=kernel_size, std=std)
    return convolve(
        img,
        kernel=kernel,
        padding=(kernel.shape[0] // 2, kernel.shape[1] // 2),
    )


if __name__ == "__main__":
    import cv2

    img_path = "./resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    kernel_size = 17
    std = 3
    out1 = cv2.GaussianBlur(img, ksize=(kernel_size, kernel_size), sigmaX=std)
    out2 = gaussian_blur(img, kernel_size=kernel_size, std=std)
    print(np.abs(out1 - out2).mean()) # 1.43
