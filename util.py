# Util functions for Quick Draw dataset experiments.

import numpy as np
import os
import sys

def load_dataset(dataset):
    """Load dataset, one of train, val, or test."""

    if dataset not in ["train", "val", "test"]:
        sys.exit("Invalid dataset type.")
    x_path = "data/split/{}_examples.npy".format(dataset)
    y_path = "data/split/{}_labels.npy".format(dataset)
    
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        sys.exit("Missing dataset files.")
    
    return np.load(x_path), np.load(y_path)

def get_category_mappings(fname = 'categories.txt'):
    """
    Arg: filename fname
    Returns: Mapping from index to category and mapping from category to index
    """
    with open(fname) as f:
        content = f.readlines()
    category_to_index = dict()
    index_to_category = []
    for i in range(len(content)):
        category_name = content[i].strip()
        index_to_category.append(category_name)
        category_to_index[category_name] = i
    return index_to_category, category_to_index

def print_statistics(num_correct, top3_correct, y_val):
    """
    Print statistics for validation set.
    """
    _, category_to_idx = get_category_mappings()
    print("="*30)
    print("FINAL RESULTS:")
    print("TOTAL ACCURACY:", sum(num_correct.values())/len(y_val))
    print("TOP 3 ACCURACY:", sum(top3_correct.values())/len(y_val))

    category_accuracy = [(cnt/np.sum(y_val == category_to_idx[category]), category)\
            for category, cnt in num_correct.items()]
    top3_category_accuracy = [(cnt/np.sum(y_val == category_to_idx[category]), category)\
            for category, cnt in top3_correct.items()]
    category_accuracy.sort(key=lambda x: -x[0])
    top3_category_accuracy.sort(key=lambda x: -x[0])

    for acc, category in category_accuracy:
        print(category, "accuracy:", acc)
    for acc, category in top3_category_accuracy:
        print(category, "top 3 accuracy:", acc)
