import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import sys
from collections import defaultdict
from util import *
from sklearn.cluster import KMeans

def compute_centroids():
    """
    Computes centroid for each data file given.
    """
    centroids = {}
    cnts = defaultdict(int)
    idx_to_category, _ = get_category_mappings()
    K = len(idx_to_category)
    train_examples = np.load("data/split/train_examples.npy")
    train_labels = np.load("data/split/train_labels.npy")

    examples_by_label = [train_examples[np.where(train_labels[:] == j)] for j in range(K)]

    #for i in range(train_examples.shape[0]):
    #    idx = train_labels[i]
    #    if train_labels[i] not in examples_by_label:
    #        examples_by_label[idx] = np.array(train_examples[i], dtype=np.float32)
    #    else:
    #        examples_by_label[idx] = np.append(examples_by_label[idx],\
    #                np.array(train_examples[i], dtype=np.float32))

    for idx, category in enumerate(idx_to_category):
        if idx%10 == 0:
            print("Done with category", idx)
        kmeans = KMeans(n_clusters=5, random_state=0).fit(examples_by_label[idx])
        clusters = kmeans.cluster_centers_
        category = idx_to_category[idx]
        for a in range(5):
            name = category + "_" + str(a)
            centroids[name] = clusters[a]
    
    return centroids

def create_centroids_dir():
    """
    Create centroids directory to save results.
    """
    try:
        os.makedirs("centroids_plus")
        os.makedirs("centroids_plus/npy")
        os.makedirs("centroids_plus/png")
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

        save_path = os.path.join("centroids_plus", category)
        plt.savefig("centroids_plus/png/"+category)
        np.save("centroids_plus/npy/"+category, centroid)
        # plt.show()

if __name__ == "__main__":
    if not os.path.isdir("data/split"):
        sys.exit("Need data directory.")
    centroids = compute_centroids()
    create_centroids_dir() 
    save_centroids(centroids)
    print("Done computing centroids!")
