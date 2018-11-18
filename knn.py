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
    num_correct, top3_correct = defaultdict(int), defaultdict(int)
    idx_to_category, _ = get_category_mappings()
    for i, x_i in enumerate(x_val):
        results = assign_to_centroids(x_i, centroids)
        category = idx_to_category[int(y_val[i])]
        if category == results[0]:
            num_correct[category] += 1
        if category in results:
            top3_correct[category] += 1
        #print("TRUE CATEGORY: {}".format(category))
        #print("ASSIGNED CATEGORIES:", results)
    return num_correct, top3_correct

if __name__ == "__main__":
    x_val, y_val = load_dataset("val") 
    centroids = load_centroids()
    num_correct, top3_correct = knn(x_val, y_val, centroids)
    print_statistics(num_correct, top3_correct, y_val)
