import numpy as np


def thicken(image, iterations=1):
    """
    Perform thickening on a binary image using NumPy by manually applying a dilation-like operation.

    Parameters:
    image (np.ndarray): Binary image (2D array with 0s and 1s).
    iterations (int): Number of iterations for thickening.

    Returns:
    np.ndarray: Thickened binary image.
    """
    # Define the 8-connectivity kernel (neighbors)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
    for _ in range(iterations):
        thickened_image = padded_image.copy()
        # Loop through each pixel (excluding the padded edges)
        for i in range(1, padded_image.shape[0] - 1):
            for j in range(1, padded_image.shape[1] - 1):
                # Extract the 3x3 neighborhood of the current pixel
                neighborhood = padded_image[i-1:i+2, j-1:j+2]
                
                # If any pixel in the 3x3 neighborhood is 1, set the center pixel to 1
                if np.any(neighborhood & kernel):
                    thickened_image[i, j] = 1
        
        padded_image = thickened_image
    
    # Remove the padding and return the result
    return padded_image[1:-1, 1:-1]

# Example binary image (0s and 1s)
binary_image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# Apply thickening
thickened_image = thicken(binary_image, iterations=2)

print("Original Image:")
print(binary_image)
print("\nThickened Image:")
print(thickened_image)
