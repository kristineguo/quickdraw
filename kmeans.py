import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        new_assignments = np.zeros(N)
        for i in range(N):
            best_dist = np.inf
            best_index = 0
            for j in range(k):
                dist = np.linalg.norm(centers[j] - features[i])
                if dist < best_dist:
                    best_dist = dist
                    best_index = j
            new_assignments[i] = best_index
        if np.array_equal(assignments, new_assignments):
            break
        new_centers = np.zeros((k, features.shape[1]))
        num_per_center = np.zeros(k)
        for i in range(N):
            new_centers[int(new_assignments[i])] += features[i]
            num_per_center[int(new_assignments[i])] += 1
        for j in range(k):
            new_centers[j] /= num_per_center[j]
        centers = new_centers.copy()
        assignments = new_assignments.copy()
            
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        ### YOUR CODE HERE
        distances = np.sqrt(((features - centers[:, np.newaxis])**2).sum(axis=2))
        new_assignments = np.argmin(distances, axis=0)
        if np.array_equal(assignments, new_assignments):
            break
        centers = np.array([features[new_assignments==j].mean(axis=0) for j in range(k)])
        assignments = new_assignments.copy()
        ### END YOUR CODE

    return assignments

