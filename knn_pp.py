import random
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import glob
import os
import sys
from collections import defaultdict
from util import *
from k_nearest_neighbor import *

def load_centroids():
    """
    Load centroids from centroids/npy directory.
    Returns map of category to its centroid, represented as numpy array.
    """
    if not os.path.isdir("centroids_plus") or not os.path.isdir("centroids_plus/npy"):
        sys.exit("Need centroids/npy directory.")

    centroid_files = glob.glob("centroids_plus/npy/*.npy")
    centroids = np.zeros((len(centroid_files), len(np.load(centroid_files[0]))))
    labels = np.zeros(len(centroid_files))
    _, cat_to_idx = get_category_mappings()
    for i in range(len(centroid_files)):
        category = os.path.splitext(os.path.basename(centroid_files[i]))[0]
        centroids[i] = np.load(centroid_files[i])
        labels[i] = cat_to_idx[category.split('_')[0]]
    return centroids, labels

if __name__ == "__main__":
    k = 10
    x_train, y_train = load_centroids()
    x_val, y_val = load_dataset("val")
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    dists = compute_distances(x_val, x_train)
    pred = predict_labels(dists, y_train, y_val, k)
    compute_scores(pred)
