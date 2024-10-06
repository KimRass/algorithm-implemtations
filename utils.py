import cv2
import numpy as np
from PIL import Image


def round_half_up(x):
    return int(x + 0.5) if x > 0 else int(x - 0.5)


def show_image(x):
    Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).show()
