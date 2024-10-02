# References:
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

import cv2
import numpy as np
from matplotlib import pyplot as plt



def perform_bin_thresholding(img, thresh, max_val, inv=False):
    """
    If `inv=False`, the output is same as `cv2.threshold(
        img, thresh=thresh, maxval=max_val, type=cv2.THRESH_BINARY,
    )[1]` and if else, `cv2.threshold(
        img, thresh=thresh, maxval=max_val, type=cv2.THRESH_BINARY_INV,
    )[1]`
    """
    if not inv:
        return np.where(img > thresh, max_val, 0).astype(np.uint8)
    else:
        return np.where(img > thresh, 0, max_val).astype(np.uint8)


def perform_ada_thresholding(img, max_val, block_size, const):
    """
    The threshold value is the mean of the neighborhood area minus the
    constant `const`.
    """
    assert block_size % 2 == 1 and block_size > 1

    padded_img = np.pad(
        # img, pad_width=block_size // 2, mode="constant", constant_values=0,
        # img, pad_width=block_size // 2, mode="edge",
        img, pad_width=block_size // 2, mode="reflect",
        # img, pad_width=block_size // 2, mode="symmetric",
    )
    new_img = np.zeros_like(img)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            loc_region = padded_img[
                row: row + block_size, col: col + block_size, ...,
            ]
            loc_thresh = np.mean(loc_region) - const
            if img[row, col, ...] > loc_thresh:
                new_img[row, col, ...] = max_val
            else:
                new_img[row, col, ...] = 0
    return new_img


const = 0
max_val = 255
block_size = 5
out1= perform_ada_thresholding(
    img, max_val=max_val, block_size=block_size, const=const,
)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = img[: 10, : 7, ...]

out2 = cv2.adaptiveThreshold(
    src=img,
    maxValue=max_val,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=block_size,
    C=const,
)
# np.array_equal(new_img, out1)
np.array_equal(new_img[3: -3], out1[2: -2])
new_img[2: -2]
out1[2: -2]
new_img[: 10, : 10]
out1[: 10, : 10]
out1


img.shape
img.ndim == 2
thresh = 127
maxval = 210
_, out1 = cv2.threshold(
    img, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY,
)
out2 = perform_bin_thresholding(img, thresh=thresh, max_val=maxval)
out1
out1.shape
np.array_equal(out1, out2)
out2
show_image(out1)
img.dtype
np.where(img > thresh, maxval, 0).astype(np.uint8)
# show_image(img)