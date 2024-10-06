import numpy as np
import math

from utils import round_half_up


def interpolate(img, size, method):
    """
    Args:
        `method` (str): The interpolation method to use when resizing the
        image.
            `"nn"`: Nearest-neighbor interpolation, where the value of the
                nearest pixel is used directly. This method results in a
                blocky, pixelated appearance.
            `"cv2_inter_nearest"`: Nearest-neighbor interpolation using the
                top-left corner of each pixel. This behaves the same as
                OpenCV's `cv2.INTER_NEAREST`, where the interpolation takes
                the value of the closest top-left pixel.
            `"bilinear"`: Bilinear interpolation, which calculates the pixel
                value using a weighted average of the four nearest pixels.
                This method provides a smoother result, especially for
                upscaled images.
    Examples:
        If `img` is:
            0 1 2
            3 4 5
        and `size=(3, 4)`, the output will vary based on the interpolation method:
        1. When `method="nn"`, the nearest-neighbor interpolation will produce:
            0 1 2 2
            3 4 5 5
            3 4 5 5
        2. When `method="cv2_inter_nearest"`, using OpenCV's nearest interpolation:
            0 0 1 2
            0 0 1 2
            3 3 4 5
        3. When `method="bilinear"`, the bilinear interpolation output will be:
            0 0 1 2
            2 2 3 4
            3 3 4 5
    """
    new_h, new_w = size
    ori_h, ori_w, *_ = img.shape
    new_img = np.zeros((new_h, new_w, *_), dtype=np.uint8)
    h_ratio = ori_h/new_h
    w_ratio = ori_w/new_w
    for new_row in range(new_h):
        for new_col in range(new_w):
            row = h_ratio*new_row
            col = w_ratio*new_col
            if method == "nn":
                nearest_row = round_half_up(row)
                nearest_col = round_half_up(col)
                val = img[nearest_row, nearest_col, ...]
            elif method == "cv2_inter_nearest":
                nearest_row = int(row)
                nearest_col = int(col)
                val = img[nearest_row, nearest_col, ...]
            elif method == "bilinear":
                row1 = math.floor(row)
                row2 = min(row1 + 1, ori_h - 1)
                col1 = math.floor(col)
                col2 = min(col1 + 1, ori_w - 1)
                p1 = img[row1, col1]
                p2 = img[row1, col2]
                p3 = img[row2, col1]
                p4 = img[row2, col2]

                if row1 == row2:
                    if col1 == col2:
                        val = p1
                    else:
                        val = p1*(col2 - col) + p2*(col - col1)
                elif col1 == col2:
                    val = p1*(row2 - row) + p3*(row - row1)
                else:
                    row1_p = p1*(col2 - col) + p2*(col - col1)
                    row2_p = p3*(col2 - col) + p4*(col - col1)
                    val = row1_p*(row2 - row) + row2_p*(row - row1)
            new_img[
                new_row, new_col, ...,
            ] = val
    return new_img.astype(np.uint8)


if __name__ == "__main__":
    import cv2

    # img = np.array(
    #     [
    #         [0., 1., 2.],
    #         [3., 4., 5.],
    #     ],
    #     dtype=np.float64,
    # )
    # size = (3, 4)
    img_path = "./resources/fox_squirrel_original.jpg"
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
    ).astype(np.uint8)
    bilinear_out2 = interpolate(img, size=size, method="bilinear")
    print(np.abs(bilinear_out1 - bilinear_out2).mean()) # 58.02
