# References:
    # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

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


def dilate_or_erode(img, kernel, num_iters=1, mode="dilate"):
    if mode == "dilate":
        func = np.max
    elif mode == "erode":
        func = np.min

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
    new_img = np.zeros_like(img)
    for row in range(padded_img.shape[0] - kernel_h + 1):
        for col in range(padded_img.shape[1] - kernel_w + 1):
            # As the kernel B is scanned over the image, we compute the
            # maximal or minimal pixel value overlapped by B and replace the
            # image pixel in the anchor point position with that maximal value.
            loc_region = padded_img[
                row: row + kernel_h, col: col + kernel_w,
            ]
            new_img[row, col] = func(loc_region * kernel)
    if num_iters == 1:
        return new_img
    return dilate_or_erode(
        new_img, kernel=kernel, num_iters=num_iters - 1, mode=mode,
    )


def dilate(img, kernel, num_iters=1):
    """
    Same as `cv2.dilate(img, kernel=kernel, iterations=num_iters)`
    """
    return dilate_or_erode(
        img, kernel=kernel, num_iters=num_iters, mode="dilate",
    )


def erode(img, kernel, num_iters=1):
    """
    Same as `cv2.erode(img, kernel=kernel, iterations=num_iters)`
    """
    return dilate_or_erode(
        img, kernel=kernel, num_iters=num_iters, mode="erode",
    )


if __name__ == "__main__":
    import cv2

    img_path = "./resources/j.webp"
    img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

    kernel = get_rect_structuring_elem((4, 5))
    num_iters = 2
    dilate_out1 = cv2.dilate(img, kernel=kernel, iterations=num_iters)
    dilate_out2 = dilate(img, kernel=kernel, num_iters=num_iters)
    print(np.array_equal(dilate_out1, dilate_out2)) # True

    erode_out1 = cv2.erode(img, kernel=kernel, iterations=num_iters)
    erode_out2 = erode(img, kernel=kernel, num_iters=num_iters)
    print(np.array_equal(erode_out1, erode_out2)) # True
