# References:
    # https://medium.com/@bosssds65/how-to-rotate-image-using-only-numpy-in-15-lines-ddc1fca93c87

import numpy as np


def get_output_size(img, angle):
    h, w, _ = img.shape
    diag = (h**2 + w**2)**0.5
    ori_angle1 = np.arctan2(h, -w)
    angle_rad = np.deg2rad(angle)
    new_angle1 = ori_angle1 - angle_rad
    new_h = int(diag * np.sin(new_angle1))
    ori_angle2 = np.arctan2(h, w)
    new_angle2 = ori_angle2 - angle_rad
    new_w = int(diag * np.cos(new_angle2))
    return new_h, new_w


def rotate(img, angle):
    angle_rad = np.deg2rad(angle)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    inv_mat = np.array([[cos, sin], [-sin, cos]])

    h, w, *_ = img.shape
    new_h, new_w = get_output_size(img, angle=angle)
    new_img = np.zeros((new_h, new_w, *_), dtype=img.dtype)
    for new_row in range(new_h):
        for new_col in range(new_w):
            out = inv_mat @ np.array(
                [[new_col - new_w//2], [new_row - new_h//2]],
            )
            ori_col, ori_row = out.flatten()
            ori_row = round(ori_row + h//2)
            ori_col = round(ori_col + w//2)
            if 0 <= ori_row < h and 0 <= ori_col < w:
                new_img[new_row, new_col] = img[ori_row, ori_col]
    return new_img


if __name__ == "__main__":
    import cv2
    import imutils

    img_path = "/Users/jongbeomkim/Desktop/workspace/numpy-image-processing/resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    angle = 44
    out1 = imutils.rotate_bound(img, angle=angle)
    out2 = rotate(img, angle=angle)
    # show_image(out1)
    # show_image(out2)
    # out1.shape, out2.shape
    print(np.abs(out1 - out2).mean())
