# https://github.com/Mr-TalhaIlyas/Conditional-Random-Fields-CRF
# The CRF tries to refine the coarse predictions made by a model by considering both the appearance of pixels (color, spatial distance, etc.) and enforcing smoothness between similar pixels.
# Key components of a CRF:
    # Unary potentials: The confidence of each pixel belonging to a particular class (usually from a model like a CNN).
    # Pairwise potentials: The cost of assigning labels to neighboring pixels, usually based on spatial proximity and similarity in color.
# Steps in Mean-Field Approximation:
    # 1. Initialize class probabilities from the unary potentials.
    # 2. Refine these probabilities by iteratively computing pairwise potentials and applying message passing.

import numpy as np
from scipy.spatial.distance import cdist

class DenseCRF:
    def __init__(self, img, num_classes, spatial_sigma=3, color_sigma=10):
        """
        Initialize the DenseCRF model.
        
        Parameters:
        - img: The input image (H, W, 3).
        - num_classes: The number of classes (e.g., for segmentation).
        - spatial_sigma: Spatial distance weight in the pairwise potential.
        - color_sigma: Color distance weight in the pairwise potential.
        """
        self.img = img
        self.H, self.W, self.C = img.shape
        self.num_classes = num_classes
        self.spatial_sigma = spatial_sigma
        self.color_sigma = color_sigma
        
        # Create pairwise features (positions and colors)
        self.positions = np.indices((self.H, self.W)).reshape(2, -1).T  # (N, 2)
        self.colors = img.reshape(-1, 3)  # (N, 3)
    
    def compute_unary_potentials(self, logits):
        """
        Compute unary potentials from logits (e.g., output of a neural network).
        Higher value means higher cost of assigning a label.
        
        Parameters:
        - logits: A (num_classes, H, W) array of logits.
        
        Returns:
        - Unary potentials as a (H * W, num_classes) array.
        """
        logits = logits.reshape(self.num_classes, -1).T  # Reshape to (N, num_classes)
        unary_potentials = -logits  # Negative log probabilities (for minimization)
        return unary_potentials
    
    def pairwise_potential(self, Q):
        """
        Compute pairwise potentials based on color and spatial features.
        
        Parameters:
        - Q: The current distribution of labels (N, num_classes).
        
        Returns:
        - A (N, num_classes) array representing the refined label probabilities.
        """
        N = self.H * self.W
        
        # Compute spatial and color differences
        spatial_diff = cdist(self.positions, self.positions, 'euclidean')  # (N, N)
        color_diff = cdist(self.colors, self.colors, 'euclidean')  # (N, N)
        
        # Apply Gaussian kernels
        spatial_kernel = np.exp(-spatial_diff ** 2 / (2 * self.spatial_sigma ** 2))
        color_kernel = np.exp(-color_diff ** 2 / (2 * self.color_sigma ** 2))
        
        # Combine spatial and color features
        pairwise_kernel = spatial_kernel * color_kernel
        
        # Message passing: pairwise_kernel * Q for each class
        refined_Q = np.zeros_like(Q)
        for c in range(self.num_classes):
            refined_Q[:, c] = pairwise_kernel.dot(Q[:, c])
        
        return refined_Q
    
    def mean_field_inference(self, unary, num_iterations=10):
        """
        Run mean-field inference to refine the unary potentials.
        
        Parameters:
        - unary: A (H * W, num_classes) array of unary potentials.
        - num_iterations: Number of mean-field iterations to perform.
        
        Returns:
        - Refined class probabilities as a (H * W, num_classes) array.
        """
        N = self.H * self.W
        
        # Initialize Q with the unary potentials
        Q = np.exp(-unary)
        Q /= Q.sum(axis=1, keepdims=True)  # Normalize to get initial probabilities
        
        for i in range(num_iterations):
            # Apply pairwise potential refinement
            pairwise_Q = self.pairwise_potential(Q)
            
            # Update Q by combining unary potentials and refined pairwise potentials
            Q = np.exp(-unary) * pairwise_Q
            Q /= Q.sum(axis=1, keepdims=True)  # Normalize again
        
        return Q
    
    def run(self, logits, num_iterations=10):
        """
        Run the full DenseCRF algorithm.
        
        Parameters:
        - logits: A (num_classes, H, W) array of logits.
        - num_iterations: Number of mean-field iterations to perform.
        
        Returns:
        - A (H, W) array of the final predicted labels.
        """
        unary = self.compute_unary_potentials(logits)
        refined_Q = self.mean_field_inference(unary, num_iterations)
        final_labels = refined_Q.argmax(axis=1).reshape(self.H, self.W)
        return final_labels
