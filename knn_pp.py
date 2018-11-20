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

    k = 12
    x_train, y_train = load_centroids()
    #y_train = np.repeat(np.arange(len(x_train)/5), 5)
    x_val, y_val = load_dataset("val")
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    #print(y_train)
    #print(labels)
    #assert np.allclose(y_train, labels)
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_val.shape)
    #print(y_val.shape)

    dists = compute_distances(x_val, x_train)

    y_test_pred = predict_labels(dists, y_train, k)

    num_test = y_val.shape[0]
    num_correct = np.sum(y_test_pred == y_val)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    '''
    k = 100
    accuracies = []
    for i in range(1, k):
      x_train, y_train = load_centroids()
      x_val, y_val = load_dataset("val")
      y_train = y_train.astype(int)
      y_val = y_val.astype(int)
      #print(x_train.shape)
      #print(y_train.shape)
      #print(x_val.shape)
      #print(y_val.shape)

      dists = compute_distances(x_val, x_train)

      y_test_pred = predict_labels(dists, y_train, i)

      num_test = y_val.shape[0]
      num_correct = np.sum(y_test_pred == y_val)
      accuracy = float(num_correct) / num_test
      print(i)
      print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
      accuracies.append(accuracy)
    plt.figure()
    plt.plot(range(k-1), accuracies)
    plt.title("Optimal k value")
    plt.xlabel("k value")
    plt.ylabel("accuracy")
    plt.savefig("best_k.png")
    '''
