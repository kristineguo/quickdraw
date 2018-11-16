import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import sys
from collections import defaultdict
from util import *

def compute_centroids():
    """
    Computes centroid for each data file given.
    """
    centroids = {}
    cnts = defaultdict(int)
    idx_to_category, _ = get_category_mappings()
    train_examples = np.load("data/split/train_examples.npy")
    train_labels = np.load("data/split/train_labels.npy")
    for i in range(train_examples.shape[0]):
        category = idx_to_category[int(train_labels[i])]
        if category not in centroids:
            centroids[category] = np.array(train_examples[i], dtype=np.float32)
        else:
            centroids[category] += train_examples[i]
        cnts[category] += 1
    for category in idx_to_category:
        centroids[category] /= cnts[category]
    return centroids

def create_centroids_dir():
    """
    Create centroids directory to save results.
    """
    try:
        os.makedirs("centroids")
        os.makedirs("centroids/npy")
        os.makedirs("centroids/png")
    except OSError:
        pass # already exists

def save_centroids(centroids):
    """
    Save all images of centroids to centroids/png.
    Save all numpy arrays of centroids to centroids/npy.
    """
    for category, centroid in centroids.items():
        plt.imshow(np.reshape(centroid, (28, 28)), cmap='gray')
        plt.title(category)

        save_path = os.path.join("centroids", category)
        plt.savefig("centroids/png/"+category)
        np.save("centroids/npy/"+category, centroid)
        # plt.show()

if __name__ == "__main__":
    if not os.path.isdir("data/split"):
        sys.exit("Need data directory.")
    centroids = compute_centroids()
    create_centroids_dir() 
    save_centroids(centroids)
    print("Done computing centroids!")
