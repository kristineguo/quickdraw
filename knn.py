import random
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import glob
import os
import sys
from collections import defaultdict
from util import *

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
        centroids[category] = np.load(path)
    return centroids

def assign_to_centroids(x_i, centroids):
    """
    Assign given example to closest centroid.
    """
    dist = [(np.dot(x_i, x_i) - 2.0*np.dot(x_i, centroid) + np.dot(centroid, centroid), category) for category, centroid in centroids.items()]
    dist.sort()
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

def print_statistics(num_correct, top3_correct, y_val):
    """
    Print statistics for validation set.
    """
    _, category_to_idx = get_category_mappings()
    print("="*30)
    print("FINAL RESULTS:")
    print("TOTAL ACCURACY:", sum(num_correct.values())/len(y_val))
    print("TOP 3 ACCURACY:", sum(top3_correct.values())/len(y_val))
    category_accuracy = [(cnt/np.sum(y_val == category_to_idx[category]), category) for category, cnt in num_correct.items()]
    top3_category_accuracy = [(cnt/np.sum(y_val == category_to_idx[category]), category) for category, cnt in top3_correct.items()]
    category_accuracy.sort(key=lambda x: -x[0])
    top3_category_accuracy.sort(key=lambda x: -x[0])
    for acc, category in category_accuracy:
        print(category, "accuracy:", acc)
    for acc, category in top3_category_accuracy:
        print(category, "top 3 accuracy:", acc)

if __name__ == "__main__":
    x_path = "data/split/val_examples.npy"
    y_path = "data/split/val_labels.npy"
    
    x_val, y_val = load_validation_set(x_path, y_path) 
    centroids = load_centroids()
    num_correct, top3_correct = knn(x_val, y_val, centroids)
    print_statistics(num_correct, top3_correct, y_val)
