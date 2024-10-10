import numpy as np

from blurring import gaussian_blur
from interpolation import interpolate


def get_gaussian_pyram(x, num_octs, num_intervals, kernel_size, init_std):
    k = 2 ** (1 / num_intervals) # Factor for sigma increase between intervals.
    pyram = []
    for oct in range(num_octs):
        oct_imgs = []
        # +3 for extra images to subtract in DoG.
        for interval in range(num_intervals + 3):
            if interval == 0:
                if oct == 0:
                    temp = x
                else:
                    before = pyram[-1][0]
                    temp = interpolate(
                        before,
                        size=(before.shape[0]//2, before.shape[1]//2),
                        method="bilinear",
                    )
            else:
                temp_std = init_std * (k ** interval)
                temp = gaussian_blur(
                    oct_imgs[-1], kernel_size=kernel_size, std=temp_std,
                )
            oct_imgs.append(temp)
        pyram.append(oct_imgs)
    return pyram


if __name__ == "__main__":
    img_path = "/Users/jongbeomkim/Desktop/workspace/image-processing-algorithms/resources/fox_squirrel_original.jpg"
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img[100: 400, 100: 400, ...]

    num_octs = 4
    num_intervals = 3
    init_std = 1.6
    kernel_size = 5
    pyram = get_gaussian_pyram(
        img,
        num_octs=num_octs,
        num_intervals=num_intervals,
        kernel_size=kernel_size,
        init_std=init_std,
    )
    for i in pyram:
        for j in i:
            show_image(j)
