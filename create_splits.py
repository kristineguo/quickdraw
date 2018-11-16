import numpy as np
import glob
from sklearn.model_selection import train_test_split
import os
import sys
from util import *

SEED = 229

def create_split_dir():
    try:
        os.makedirs("data/split")
    except OSError:
        pass # already exists

if __name__ == "__main__":
    data_files = glob.glob("data/*.npy")
    train_examples, train_labels, val_examples, val_labels, test_examples, test_labels = [], [], [], [], [], []
    _, category_to_idx = get_category_mappings()
    create_split_dir()
    for path in data_files:
        all_examples = np.load(path)
        category = os.path.splitext(os.path.basename(path))[0]
        idx = category_to_idx[category]
        _, sampled_X, _, sampled_y = train_test_split(all_examples, np.ones(all_examples.shape[0]) * idx, test_size=0.01, random_state=SEED)
        train_X, val_test_X, train_y, val_test_y = train_test_split(sampled_X, np.ones(sampled_X.shape[0]) * idx, test_size=0.3, random_state=SEED)
        val_X, test_X, val_y, test_y = train_test_split(val_test_X, np.ones(val_test_X.shape[0]) * idx, test_size=0.5, random_state=SEED)
        train_examples.append(train_X)
        train_labels.append(train_y)
        val_examples.append(val_X)
        val_labels.append(val_y)
        test_examples.append(test_X)
        test_labels.append(test_y)
    train_examples = np.concatenate(train_examples)
    train_labels = np.concatenate(train_labels)
    val_examples = np.concatenate(val_examples)
    val_labels = np.concatenate(val_labels)
    test_examples = np.concatenate(test_examples)
    test_labels = np.concatenate(test_labels)
    print("number of training examples", train_examples.shape[0])
    print("number of validation examples", val_examples.shape[0])
    print("number of test examples", test_examples.shape[0])
    np.save("data/split/train_examples", train_examples)
    np.save("data/split/train_labels", train_labels)
    np.save("data/split/val_examples", val_examples)
    np.save("data/split/val_labels", val_labels)
    np.save("data/split/test_examples", test_examples)
    np.save("data/split/test_labels", test_labels)
