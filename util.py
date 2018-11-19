# Util functions for Quick Draw dataset experiments.
import numpy as np
import os
import sys

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

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

def compute_scores(pred):
    actual, predicted, per_category_mapk = [], [], []
    for category, guesses in pred.items():
        cur_actual, cur_predicted = [], []
        occ = Counter()
        for guess in guesses:
            cur_actual.append([category])
            cur_predicted.append(guess)
            for cat in guess:
                occ[cat] += 1
        per_category_mapk.append((category, mapk(cur_actual, cur_predicted), occ.most_common(3)))
        actual += cur_actual
        predicted += cur_predicted
    per_category_mapk.sort(key=lambda x: -x[1])

    print("="*30)
    print("MAPK@3 SCORE:", mapk(actual, predicted))
    for category, acc, guess in per_category_mapk:
        print(category, "MAPK@3:", acc, "common guesses:", guess)
