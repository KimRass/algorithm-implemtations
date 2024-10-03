# References:
    # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

import cv2
import numpy as np


def get_rect_structuring_elem(kernel_size):
    assert isinstance(kernel_size, int) or (
        isinstance(kernel_size, tuple) and len(kernel_size) == 2
    )
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    return np.full(
        (kernel_size[0], kernel_size[1]), fill_value=1, dtype=np.uint8,
    )


def thicken(img, kernel, num_iters=1):
    new_img = np.zeros_like(img)
    for _ in range(num_iters):
        new_img = thicken(img, kernel=kernel, num_iters=num_iters - 1)

    kernel_h, kernel_w = kernel.shape    
    padded_img = np.pad(
        img,
        pad_width=(
            (kernel_h // 2, (kernel_h - 1) // 2),
            (kernel_w // 2, (kernel_w - 1) // 2),
        ),
        mode="constant",
        constant_values=0,
    )
    for row in range(padded_img.shape[0] - kernel_h + 1):
        for col in range(padded_img.shape[1] - kernel_w + 1):
            # As the kernel B is scanned over the image, we compute the
            # maximal pixel value overlapped by B and replace the image
            # pixel in the anchor point position with that maximal value.
            loc_region = padded_img[
                row: row + kernel_h, col: col + kernel_w,
            ]
            new_img[row, col] = np.max(loc_region * kernel)
    return new_img


if __name__ == "__main__":
    img_path = "/Users/jongbeomkim/Desktop/workspace/numpy-image-processing/resources/j.webp"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = get_rect_structuring_elem((4, 5))
    num_iters = 1
    out1 = thicken(img, kernel=kernel, num_iters=num_iters)
    out2 = cv2.dilate(img, kernel=kernel, iterations=num_iters)
    print(np.array_equal(out1, out2)) # `True`
