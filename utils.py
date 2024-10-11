import cv2
import numpy as np
from PIL import Image


def round_half_up(x):
    return int(x + 0.5) if x > 0 else int(x - 0.5)


def show_image(x):
    Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).show()


def to_grayscale(img):
    gray_img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    return gray_img
