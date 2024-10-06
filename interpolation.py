import numpy as np
import math

from utils import round_half_up


def interpolate(img, size, method):
    """
    Args:
        method (str): Interpolation method. Options are:
            `"nn"`:
            `"cv2_inter_nearest"`: Nearest-neighbor interpolation using the
            top-left corner of the pixel. Same as `cv2.INTER_NEAREST`.
    Example:
        If `img` is:
            0 1 2
            3 4 5
        When `size=(3, 4)` and `method="nn"`, the output will be:
            0 1 2 2
            3 4 5 5
            3 4 5 5
        When `size=(3, 4)` and `method="cv2_inter_nearest"`, the output will
        be:
            0 0 1 2
            0 0 1 2
            # 3 3 4 5
    """
    new_h, new_w = size
    ori_h, ori_w, *_ = img.shape
    if method in ["nn", "cv2_inter_nearest", "cv2_inter_nearest_exact"]:
        dtype = img.dtype
    else:
        dtype = np.float64
    if img.ndim == 2:
        new_img = np.zeros((new_h, new_w), dtype=dtype)
    elif img.ndim == 3:
        new_img = np.zeros((new_h, new_w, img.shape[2]), dtype=dtype)
    h_ratio = ori_h/new_h
    w_ratio = ori_w/new_w
    for new_row in range(new_h):
        for new_col in range(new_w):
            row = h_ratio*new_row
            col = w_ratio*new_col
            if method == "nn":
                nearest_row = round_half_up(row)
                nearest_col = round_half_up(col)
                new_img[new_row, new_col, ...] = img[
                    nearest_row, nearest_col, ...,
                ]
            elif method == "cv2_inter_nearest":
                nearest_row = int(row)
                nearest_col = int(col)
                new_img[new_row, new_col, ...] = img[
                    nearest_row, nearest_col, ...,
                ]
            elif method == "bilinear":
                row1 = math.floor(row)
                row2 = min(row1 + 1, ori_h - 1)
                if row1 == row2:
                    break
                col1 = math.floor(col)
                col2 = min(col1 + 1, ori_w - 1)
                p1 = img[row1, col1]
                p2 = img[row1, col2]
                p3 = img[row2, col1]
                p4 = img[row2, col2]
                row1_p = p1*(col2 - col) + p2*(col - col1)
                row2_p = p3*(col2 - col) + p4*(col - col1)
                new_img[
                    new_row, new_col, ...,
                ] = row1_p*(row2 - row) + row2_p*(row - row1)
    return new_img.astype(np.uint8)


def interpolate(img, size):
    new_h, new_w = size
    ori_h, ori_w, *_ = img.shape
    if img.ndim == 2:
        new_img = np.zeros((new_h, new_w), dtype=np.uint8)
    elif img.ndim == 3:
        new_img = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
    h_ratio = ori_h/new_h
    w_ratio = ori_w/new_w
    for new_row in range(new_h):
        for new_col in range(new_w):
            row = h_ratio*new_row
            col = w_ratio*new_col

            row1 = math.floor(row)
            row2 = min(row1 + 1, ori_h - 1)
            if row1 == row2:
                break
            col1 = math.floor(col)
            col2 = min(col1 + 1, ori_w - 1)
            p1 = img[row1, col1]
            p2 = img[row1, col2]
            p3 = img[row2, col1]
            p4 = img[row2, col2]
            row1_p = p1*(col2 - col) + p2*(col - col1)
            row2_p = p3*(col2 - col) + p4*(col - col1)
            new_img[
                new_row, new_col, ...,
            ] = row1_p*(row2 - row) + row2_p*(row - row1)
    return new_img.astype(np.uint8)


if __name__ == "__main__":
    import cv2

    img = np.array(
        [
            [0., 1., 2.],
            [3., 4., 5.],
        ],
        dtype=np.float64,
    )
    size = (3, 4)
    img_path = "/Users/jongbeomkim/Desktop/workspace/numpy-image-processing/resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    size = (325, 900)

    nn_out1 = cv2.resize(
        img, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST,
    )
    nn_out2 = interpolate(img, size=size, method="nn")
    nn_out3 = interpolate(img, size=size, method="cv2_inter_nearest")
    print(np.abs(nn_out1 - nn_out2).mean()) # 73.61
    print(np.abs(nn_out1 - nn_out3).mean()) # 0.0

    bilinear_out1 = cv2.resize(
        img, dsize=(size[1], size[0]), interpolation=cv2.INTER_LINEAR,
    )
    bilinear_out1.astype(np.uint8)
    bilinear_out2 = interpolate(img, size=size, method="bilinear")
    bilinear_out1
    bilinear_out2
    print(np.abs(bilinear_out1 - bilinear_out2).mean()) # 58.18
