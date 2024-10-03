# References:
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

import cv2
import numpy as np


def get_rect_structuring_elem(kernel_size):
    return np.full((kernel_size, kernel_size), fill_value=1, dtype=np.uint8)


def thicken(image, kernel, iterations=1):
    padded_img = np.pad(image, pad_width=1, mode="constant", constant_values=0)
    for _ in range(iterations):
        new_img = padded_img.copy()
        for i in range(1, padded_img.shape[0] - 1):
            for j in range(1, padded_img.shape[1] - 1):
                # Extract the 3x3 neighborhood of the current pixel
                neighborhood = padded_img[i - 1: i + 2, j - 1: j + 2]
                # If any pixel in the 3x3 neighborhood is 1, set the center pixel to 1
                if np.any(neighborhood & kernel):
                    new_img[i, j] = 1
        padded_img = new_img
    return padded_img[1:-1, 1:-1]


if __name__ == "__main__":
    img_path = "/Users/jongbeomkim/Desktop/workspace/numpy-image-processing/resources/j.webp"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel = get_rect_structuring_elem(2)
    bin_img = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    new_img = thicken(bin_img, iterations=1)
    new_img
    show_image(img)
        
    kernel_size = 1
    dilation = cv2.dilate(img, kernel, iterations = 1)
    show_image(dilation)
