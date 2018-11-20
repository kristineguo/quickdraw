import numpy as np
from math import log10, floor

def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

def compute_distances(X1, X2):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    dists = np.zeros((M, N))
    #print("Computing Distances")
    #print(X1.shape)
    #print(X2.shape)
    #dists = np.sqrt(((X2 - X1[:, np.newaxis])**2).sum(axis=2))
    
    print("Computing Distances")
    benchmark = int(round_to_1(M)//10)
    for i in range(len(X1)):
      for j in range(len(X2)):
        dists[i,j] = np.linalg.norm(X1[i] - X2[j])
      if i % benchmark == 0:
        print(str(i//benchmark)+"0% complete")
    print("Distances Computed")


    '''
    a_2 = np.sum(np.square(X1), axis = 1)
    ab2 = 2*np.dot(X1, X2.T)
    b_2 = np.sum(np.square(X2), axis = 1)
    dists = np.sqrt(np.abs(a_2[:,np.newaxis]-ab2+b_2[np.newaxis,:]))
    '''

    #print("Distances computed!")

    assert dists.shape == (M, N), "dists should have shape (M, N), got %s" % dists.shape

    return dists


def predict_labels(dists, y_train, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test, num_train = dists.shape
    y_pred = np.zeros(num_test, dtype=np.int)

    for i in range(num_test):
        closest_y = y_train[np.argpartition(dists[i], k)[:k]]
        occur = np.bincount(closest_y)
        max_occur = np.max(occur)
        y_pred[i] = np.min(np.where(occur == max_occur))
        #y_pred[i] = np.argmax(np.bincount(closest_y))

    return y_pred


def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    jeturns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)

    """
    assert X_train.shape[0] == y_train.shape[0]

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size

    X_trains = np.zeros((num_folds, training_size, X_train.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, X_train.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    for i in range(num_folds):
        X_trains[i] = np.concatenate((X_train[:i*validation_size], X_train[(i+1)*validation_size:]))
        X_vals[i] = X_train[i*validation_size:(i+1)*validation_size]
        y_trains[i] = np.concatenate((y_train[:i*validation_size], y_train[(i+1)*validation_size:]))
        y_vals[i] = y_train[i*validation_size:(i+1)*validation_size]

    return X_trains, y_trains, X_vals, y_vals
