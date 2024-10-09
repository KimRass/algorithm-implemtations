import numpy as np

def ransac(data, model, n, k, t, d):
    """
    :param data: Input data points as a 2D numpy array of shape (N, 2)
    :param model: Function that fits a model to a subset of data points
    :param n: Minimum number of data points required to fit the model
    :param k: Maximum number of iterations for the algorithm
    :param t: Threshold to determine whether a point fits the model
    :param d: Minimum number of data points required to assert that a model is good
    :return: Best model parameters found
    """
    best_model = None
    best_inliers = []
    best_error = float('inf')

    for _ in range(k):
        # Randomly select a subset of n points
        maybe_inliers = data[np.random.choice(data.shape[0], n, replace=False)]
        
        # Fit the model to these points
        maybe_model = model(maybe_inliers)
        
        # Compute inliers for this model
        inliers = []
        for point in data:
            if np.abs(np.dot(maybe_model[:2], point) + maybe_model[2]) < t:
                inliers.append(point)
        
        # Check if the number of inliers is sufficient
        if len(inliers) > d:
            inliers = np.array(inliers)
            # Refit the model to all inliers
            better_model = model(inliers)
            
            # Calculate error (sum of residuals)
            error = np.sum(np.abs(np.dot(inliers, better_model[:2]) + better_model[2]))
            
            # Update the best model if it has a lower error
            if error < best_error:
                best_model = better_model
                best_inliers = inliers
                best_error = error

    return best_model, best_inliers


def fit_line(data):
    """
    Fits a 2D line to a set of points.
    :param data: Input data points as a 2D numpy array of shape (N, 2)
    :return: Parameters of the line in the form (a, b, c) for ax + by + c = 0
    """
    x = data[:, 0]
    y = data[:, 1]
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Line equation: y = mx + c -> mx - y + c = 0 -> (m, -1, c)
    return np.array([m, -1, c])


# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    
    # Generate synthetic data (line with some noise and outliers)
    n_points = 100
    x = np.linspace(0, 10, n_points)
    y = 2 * x + 1 + np.random.normal(0, 1, n_points)  # y = 2x + 1 with noise
    
    # Add some outliers
    outliers = np.random.rand(20, 2) * 10
    data = np.vstack([np.column_stack((x, y)), outliers])

    # RANSAC parameters
    n = 2          # Minimum number of points to fit the model
    k = 100        # Number of iterations
    t = 1.0        # Threshold to consider a point as an inlier
    d = 50         # Minimum number of inliers to accept the model
    
    best_model, best_inliers = ransac(data, fit_line, n, k, t, d)

    print(f"Best model: {best_model}")
