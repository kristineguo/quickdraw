import random
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import glob
import os
import sys

def load_validation_set(x_path, y_path):
    """
    Load validation set for evaluation.
    """
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        sys.exit("Missing validation dataset.")
    x_val, y_val = np.load(x_path), np.load(y_path)
    return x_val, y_val

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
        centroids[path] = np.load(path)
    return centroids

def assign_to_centroid(x_i, centroids):
    """
    Assign given example to closest centroid.
    """
    minCategory = "n/a"
    minValue = float('inf')
    for category, centroid in centroids.items():
        d = np.dot(x_i, x_i) - 2.0*np.dot(x_i, centroid) + np.dot(centroid, centroid)
        if d < minValue:
            minValue = d
            minCategory = category
    return [minCategory]

def knn(x_val, y_val, centroids):
    num_correct = 0
    for i, x_i in enumerate(x_val):
        results = assign_to_centroids(x_i, centroids)
        if y_val[i] in results:
            num_correct += 1
        print("TRUE CATEGORY: {}".format(y_val[i]))
        print("ASSIGNED CATEGORIES:", results)
    return num_correct

def print_statistics(num_correct, num_total):
    """
    Print statistics for validation set.
    """
    print("="*30)
    print("FINAL RESULTS:")
    print("NUM CORRECT:", num_correct)
    print("NUM INCORRECT:", num_total - num_correct)

if __name__ == "__main__":
    x_path = "validation.npy" # TODO: Update.
    y_path = "validation_labels.npy" #TODO: Update.
    
    x_val, y_val = load_validation_set(x_path, y_path) 
    centroids = load_centroids()
    num_correct = knn(x_val, y_val, centroids)
    print_statistics(num_correct, len(x_val))
