# References:
    # https://en.wikipedia.org/wiki/Difference_of_Gaussians
    # https://velog.io/@kowoonho/SIFT-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98

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
                    oct_img = x
                else:
                    prev_oct_img = pyram[-1][0]
                    oct_img = interpolate(
                        prev_oct_img,
                        size=(
                            prev_oct_img.shape[0]//2,
                            prev_oct_img.shape[1]//2,
                        ),
                        method="bilinear",
                    )
            else:
                # The blurred images are obtained by convolving the original
                # images with Gaussian kernels having standard deviations.
                oct_std = init_std * (k ** interval)
                oct_img = gaussian_blur(
                    oct_imgs[-1], kernel_size=kernel_size, std=oct_std,
                )
            oct_imgs.append(oct_img)
        pyram.append(oct_imgs)
    return pyram


def get_dog_pyram(gaussian_pyram):
    dogs = []
    for oct in gaussian_pyram:
        dog_imgs = []
        for idx in range(len(oct) - 1):
            # The subtraction of one Gaussian blurred version of an original
            # image from another, less blurred version of the original.
            dog_imgs.append(oct[idx + 1] - oct[idx])
        dogs.append(dog_imgs)
    return dogs


def get_keypoints(dog_pyram):
    keypoints = []
    for oct in dog_pyram:
        stacked_dog = np.stack(oct)
        for idx in range(1, len(oct) - 1):
            temp = stacked_dog[idx - 1: idx + 2]

            for row in range(1, temp.shape[1] - 1):
                for col in range(1, temp.shape[2] - 1):
                    patch = temp[:, row - 1: row + 2, col - 1: col + 2]
                    center = patch[1, 1, 1]
                    
                    patch1 = patch.copy()
                    patch1[1, 1, 1] += 1
                    patch2 = patch.copy()
                    patch2[1, 1, 1] -= 1
                    # print(np.array_equal(patch, patch1), np.array_equal(patch, patch2))
                    if np.all(center > patch1) or np.all(center < patch2):
                        keypoints.append((idx, row, col))
    # print(len(keypoints))
    return keypoints


if __name__ == "__main__":
    from utils import show_image

    img_path = "/Users/jongbeomkim/Desktop/workspace/image-processing-algorithms/resources/fox_squirrel_original.jpg"
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img[100: 400, 100: 400, ...]

    num_octs = 4
    num_intervals = 3
    init_std = 1.6
    kernel_size = 5
    gaussian_pyram = get_gaussian_pyram(
        img,
        num_octs=num_octs,
        num_intervals=num_intervals,
        kernel_size=kernel_size,
        init_std=init_std,
    )
    dog_pyram = get_dog_pyram(gaussian_pyram)
    keypoints = get_keypoints(dog_pyram)
    show_image(dog_pyram[1][1])
    show_image(dog_pyram[0][1])
