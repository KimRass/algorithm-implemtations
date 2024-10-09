# References:
    # https://en.wikipedia.org/wiki/Sobel_operator
    # https://gaussian37.github.io/vision-concept-edge_detection/

import numpy as np

x = np.zeros((3, 4), dtype=np.uint8)
x[2, 0] = 300
x

if __name__ == "__main__":
    import cv2

    img_path = "/Users/jongbeomkim/Desktop/workspace/image-processing-algorithms/resources/fox_squirrel_original.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    kernel_size = 5
    std = 3
    out1 = cv2.GaussianBlur(img, ksize=(kernel_size, kernel_size), sigmaX=std)
    out2 = gaussian_blur(img, kernel_size=kernel_size, std=std)
    # show_image(out1)
    # show_image(out2)
    print(np.abs(out1 - out2).mean())
    # out1[: 5, : 5, 0]
    # out2[: 5, : 5, 0]
