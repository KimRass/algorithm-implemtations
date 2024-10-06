# References:
    # https://medium.com/@bosssds65/how-to-rotate-image-using-only-numpy-in-15-lines-ddc1fca93c87

import numpy as np

from utils import round_half_up


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


def rotate1(img, angle, method="nn"):
    angle_rad = np.deg2rad(angle)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    h, w, *_ = img.shape
    new_h, new_w = get_output_size(img, angle=angle)
    new_img = np.zeros((new_h, new_w, *_), dtype=img.dtype)
    for new_row in range(new_h):
        for new_col in range(new_w):
            a = new_col - new_w//2
            b = new_row - new_h//2
            if method == "nn":
                ori_row = round_half_up(-sin*a + cos*b) + h//2
                ori_col = round_half_up(cos*a + sin*b) + w//2
            if 0 <= ori_row < h and 0 <= ori_col < w:
                new_img[new_row, new_col] = img[ori_row, ori_col]
    return new_img


def rotate2(img, angle):
    angle_rad = np.deg2rad(angle)
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    h, w, *_ = img.shape
    new_h, new_w = get_output_size(img, angle=angle)
    new_img = np.zeros((new_h, new_w, *_), dtype=img.dtype)
    for ori_row in range(h):
        for ori_col in range(w):
            a = ori_col - w//2
            b = ori_row - h//2
            new_row = round_half_up(sin*a + cos*b + new_h//2)
            new_col = round_half_up(cos*a - sin*b + new_w//2)
            if 0 <= new_row < new_h and 0 <= new_col < new_w:
                new_img[new_row, new_col] = img[ori_row, ori_col]
    return new_img


if __name__ == "__main__":
    import cv2
    import imutils

    img_path = "./resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    angle = 44
    method = "nn"
    out1 = imutils.rotate_bound(img, angle=angle)
    out2 = rotate1(img, angle=angle, method=method)
    print(np.abs(out1 - out2).mean()) # 37.27
    out3 = rotate2(img, angle=angle)
    print(np.abs(out1 - out3).mean()) # 44.38
