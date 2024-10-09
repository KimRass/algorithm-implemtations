import numpy as np
import einops


def add_gaussian_noise(img, std):
    """Add Gaussian noise.

    Args:
        img (Numpy array): Input image of range $[0, 1]$, dtype `np.float64`.
        std (float): Noise scale measured in range $[0, 255]$.

    Returns:
        (Numpy array): Returned noisy image of range $[0, 1]$, dtype
        `np.float64`.
    """
    h, w, c = img.shape
    noise = np.random.randn(h, w) * (std / 255)
    noise = einops.repeat(noise, pattern="h w -> h w c", c=c)
    return np.clip(img + noise, 0, 1)
