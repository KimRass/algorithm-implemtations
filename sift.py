import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def gaussian_pyramid(image, num_octaves, num_intervals, sigma):
    """
    Build a Gaussian Pyramid with multiple octaves.
    
    :param image: Input image (grayscale).
    :param num_octaves: Number of octaves in the pyramid.
    :param num_intervals: Number of intervals per octave.
    :param sigma: Initial sigma for Gaussian blurring.
    :return: List of Gaussian blurred images for each octave.
    """
    pyramid = []
    k = 2 ** (1 / num_intervals)  # Factor for sigma increase between intervals
    
    for octave in range(num_octaves):
        octave_images = []
        for interval in range(num_intervals + 3):  # +3 for extra images to subtract in DoG
            if octave == 0 and interval == 0:
                blurred_image = image
            elif interval == 0:
                # Downsample previous octave
                blurred_image = cv2.resize(pyramid[-1][-3], (image.shape[1] // 2, image.shape[0] // 2))
            else:
                sigma_i = sigma * (k ** interval)
                blurred_image = gaussian_filter(octave_images[-1], sigma=sigma_i)

            octave_images.append(blurred_image)
        pyramid.append(octave_images)
    return pyramid


def difference_of_gaussians(gaussian_pyramid):
    """
    Compute the Difference of Gaussians (DoG) from the Gaussian Pyramid.
    
    :param gaussian_pyramid: Gaussian pyramid generated from the input image.
    :return: DoG pyramid.
    """
    dog_pyramid = []
    for octave_images in gaussian_pyramid:
        dog_images = []
        for i in range(1, len(octave_images)):
            dog_image = octave_images[i] - octave_images[i - 1]
            dog_images.append(dog_image)
        dog_pyramid.append(dog_images)
    return dog_pyramid


def find_keypoints(dog_pyramid, contrast_threshold=0.04):
    """
    Detect keypoints by finding local extrema in the DoG pyramid.
    
    :param dog_pyramid: Difference of Gaussians pyramid.
    :param contrast_threshold: Threshold to filter weak keypoints.
    :return: List of keypoints as (x, y, octave, scale).
    """
    keypoints = []
    for octave_index, dog_images in enumerate(dog_pyramid):
        for scale_index in range(1, len(dog_images) - 1):
            dog_previous = dog_images[scale_index - 1]
            dog_current = dog_images[scale_index]
            dog_next = dog_images[scale_index + 1]

            # Iterate over all pixels and find local extrema
            for i in range(1, dog_current.shape[0] - 1):
                for j in range(1, dog_current.shape[1] - 1):
                    patch = dog_current[i - 1:i + 2, j - 1:j + 2]
                    if np.abs(dog_current[i, j]) > contrast_threshold and (
                            np.all(dog_current[i, j] > patch) or np.all(dog_current[i, j] < patch)):
                        keypoints.append((i, j, octave_index, scale_index))
    return keypoints


# Example usage
if __name__ == "__main__":
    image = cv2.imread("input_image.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Parameters
    num_octaves = 4
    num_intervals = 3
    initial_sigma = 1.6
    
    # Step 1: Create Gaussian Pyramid
    gaussian_pyr = gaussian_pyramid(image, num_octaves=num_octaves, num_intervals=num_intervals, sigma=initial_sigma)
    
    # Step 2: Compute Difference of Gaussians (DoG)
    dog_pyr = difference_of_gaussians(gaussian_pyr)
    
    # Step 3: Detect Keypoints in DoG
    keypoints = find_keypoints(dog_pyr)
    
    print(f"Detected {len(keypoints)} keypoints")
