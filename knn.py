import random
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import glob
import os
import sys
from collections import defaultdict
from util import *

def load_centroids():
    """
    Load centroids from centroids/npy directory.
    Returns map of category to its centroid, represented as numpy array.
    """
    if not os.path.isdir("centroids") or not os.path.isdir("centroids/npy"):
        sys.exit("Need centroids/npy directory.")

    centroids = {}
    centroid_files = glob.glob("centroids/npy/*.npy")
    for path in centroid_files:
        category = os.path.splitext(os.path.basename(path))[0]
        centroids[category] = np.load(path)
    return centroids

def assign_to_centroids(x_i, centroids):
    """
    Assign given example to closest centroid.
    """
    def get_dist(x_i, centroid):
        return np.dot(x_i, x_i) - 2.0*np.dot(x_i, centroid)\
                + np.dot(centroid, centroid)

    dist = sorted([(get_dist(x_i, centroid), category)\
            for category, centroid in centroids.items()])
    return [category for _, category in dist[:3]]

def knn(x_val, y_val, centroids):
    pred = defaultdict(list)
    idx_to_category, _ = get_category_mappings()
    for i, x_i in enumerate(x_val):
        results = assign_to_centroids(x_i, centroids)
        category = idx_to_category[int(y_val[i])]
        pred[category].append(results)
    return pred

if __name__ == "__main__":
    x_val, y_val = load_dataset("val")
    centroids = load_centroids()
    pred = knn(x_val, y_val, centroids)
    compute_scores(pred)
