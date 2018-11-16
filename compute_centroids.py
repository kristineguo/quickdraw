import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import sys

def compute_centroids(data_files):
    """
    Computes centroid for each data file given.
    """
    centroids = {}
    for path in data_files:
        img_array = np.load(path)
        category = os.path.splitext(os.path.basename(path))[0]
        centroids[category] = np.sum(img_array, axis=0, dtype=float) / len(img_array)
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
    if not os.path.isdir("data"):
        sys.exit("Need data directory.")
    
    data_files = glob.glob("data/*.npy")
    centroids = compute_centroids(data_files)
    create_centroids_dir() 
    save_centroids(centroids)
    print("Done computing centroids!")
