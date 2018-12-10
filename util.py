# Util functions for Quick Draw dataset experiments.
import numpy as np
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
from confusion_matrix import plot_confusion_matrix
from collections import defaultdict

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

    if dataset not in ["train", "val", "test", "train_normalized", "val_normalized"]:
        sys.exit("Invalid dataset type.")
    x_path = "data/split/{}_examples.npy".format(dataset)
    y_path = "data/split/{}_labels.npy".format(dataset.split('_')[0])
    
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        sys.exit("Missing dataset files.")
    
    return np.load(x_path), np.asarray(np.load(y_path), dtype=np.int32)

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

def get_groupings(per_category_mapk, num=2):
    '''Create groupings of categories based on common guesses.
    Num is the number of guesses to take into account.'''
    index_to_category, _ = get_category_mappings()
    assignments = {}
    groupings = defaultdict(set)

    for category, acc, guess in per_category_mapk:
        assignment = -1
        if category not in assignments:
            for cat, _ in guess[:num]:
                if cat in assignments:
                    assignment = assignments[cat]
                    break
            if assignment == -1:
                assignment = len(groupings)
            assignments[category] = assignment
            groupings[assignment].add(category)
        else:
            assignment = assignments[category]
        for cat, _ in guess[:num]:
            if cat not in assignments:
                assignments[cat] = assignment
                groupings[assignment].add(cat)
    print(assignments)
    for i, grouping in groupings.items():
        print("="*10)
        print("GROUP", i)
        for g in grouping:
            print("\t", index_to_category[g])


def plot_accuracies(acc_vals):
    '''Plot histogram of MAP@3 Values'''
    plt.figure("KNN Accuracies")
    plt.hist(acc_vals, 19)
    plt.title("MAP@3 Accuracy Distribution for KNN (K-Means++, Weighted)")
    plt.xlabel("MAP@3 Accuracy")
    plt.ylabel("Number of Categories with Given Accuracy")
    plt.savefig("KNN_Accuracies")

def compute_scores(pred, verbose=False):
    actual, predicted, per_category_mapk = [], [], []
    total_accuracy = 0.0
    for category, guesses in pred.items():
        cur_actual, cur_predicted = [], []
        occ = Counter()
        for guess in guesses:
            cur_actual.append([category])
            cur_predicted.append(guess)
            for cat in guess:
                occ[cat] += 1
            if guess[0] == category:
                total_accuracy += 1
        per_category_mapk.append((category, mapk(cur_actual, cur_predicted), occ.most_common(3)))
        actual += cur_actual
        predicted += cur_predicted
    per_category_mapk.sort(key=lambda x: -x[1])

    # Get MAP@3 scores for all categories
    index_to_category, _ = get_category_mappings()
    acc_vals = []
    for category, acc, guess in per_category_mapk:
    #    print(index_to_category[category], "MAPK@3:", acc, "common guesses:",\
    #            [(index_to_category[g[0]], g[1]) for g in guess])
        acc_vals.append(acc)
    
    # Get groupings of categories based on common guesses
    # get_groupings(per_category_mapk)

    # Plot histogram of accuracies
    # plot_accuracies(acc_vals)

    if verbose:
        print("="*30)
        print("MAPK@3:", mapk(actual, predicted))
        print("TOTAL ACCURACY:", total_accuracy/len(actual))
        for category, acc, guess in per_category_mapk:
            print(category, "MAPK@3:", acc, "common guesses:", guess)

    return mapk(actual, predicted), total_accuracy/len(actual)
