import numpy as np


def interpolate(img, size, method):
    """
    Args:
        method (str): Interpolation method. Options are:
            "top_left_nn": Nearest-neighbor interpolation using the top-left
                corner of the pixel. Same as `cv2.INTER_NEAREST`.
            "center_middle_nn": Nearest-neighbor interpolation using the center
                of the pixel. Same as `cv2.INTER_NEAREST_EXACT`.
    Example:
        If `img` is:
            [[0, 1, 2],
             [3, 4, 5]]
        When `size=(3, 4)` and `method="top_left_nn"`, the output will be:
            [[0, 0, 1, 2],
             [0, 0, 1, 2],
             [3, 3, 4, 5]]
        When `size=(3, 4)` and `method="center_middle_nn"`, the output will be:
            [[0, 1, 1, 2],
             [3, 4, 4, 5],
             [3, 4, 4, 5]]
    """
    method = "align_bilinear"
    new_h, new_w = size
    ori_h, ori_w, *_ = img.shape
    if method in ["top_left_nn", "center_middle_nn"]:
        dtype = img.dtype
    else:
        dtype = np.float64
    if img.ndim == 2:
        new_img = np.zeros((new_h, new_w), dtype=dtype)
    elif img.ndim == 3:
        new_img = np.zeros((new_h, new_w, img.shape[2]), dtype=dtype)
    for i in range(new_h):
        for j in range(new_w):
            if method == "top_left_nn":
                nearest_i = int((ori_h / new_h) * i)
                nearest_j = int((ori_w / new_w) * j)
                new_img[i, j, ...] = img[nearest_i, nearest_j, ...]
            elif method == "center_middle_nn":
                nearest_i = int((ori_h / new_h) * (i + 1 / 2))
                nearest_j = int((ori_w / new_w) * (j + 1 / 2))
                new_img[i, j, ...] = img[nearest_i, nearest_j, ...]
            elif method == "align_bilinear":
                l = int((ori_w - 1) / (new_w - 1) * j)
                t = int((ori_h - 1) / (new_h - 1) * i)
                r = min(ori_w - 1, l + 1)
                b = min(ori_h - 1, t + 1)
                dist_l = (ori_w - 1) / (new_w - 1) * j - l
                dist_t = (ori_h - 1) / (new_h - 1) * i - t
                dist_r = l + 1 - (ori_w - 1) / (new_w - 1) * j
                dist_b = t + 1 - (ori_h - 1) / (new_h - 1) * i
                # t, l, t, r
                # img.shape
                center_top = img[t, l] * dist_r +  img[t, r] * dist_l
                center_bottom = img[b, l] * dist_r +  img[b, r] * dist_l
                center_middle = center_top * dist_b + center_bottom * dist_t
                new_img[i, j, ...] = center_middle
    new_img
    return new_img


if __name__ == "__main__":
    import cv2

    img = np.array(
        [
            [0., 1., 2.],
            [3., 4., 5.],
        ],
        dtype=np.float64,
    )
    cv2.resize(img, dsize=(4, 3), interpolation=cv2.INTER_LINEAR)
    cv2.resize(img, dsize=(4, 3), interpolation=cv2.INTER_LINEAR_EXACT)
    # img_path = "/Users/jongbeomkim/Documents/datasets/flickr8k/Images/90011335_cfdf9674c2.jpg"
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # interpolate(img, size=(3, 4), method="center_middle_nn")
    0.625 * 2
    np.array_equal(
        interpolate(img, size=(3, 4), method="top_left_nn"),
    )
    np.array_equal(
        interpolate(img, size=(3, 4), method="center_middle_nn"),
        cv2.resize(img, dsize=(4, 3), interpolation=cv2.INTER_NEAREST_EXACT),
    )
    # Image.fromarray(img).show()

    # new_img = nearest_interpolate(img, size=(400, 900))
    # Image.fromarray(new_img).show()