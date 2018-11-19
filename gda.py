import numpy as np
from util import *
import operator
from collections import Counter, defaultdict

def fit(x, y):
    """Fit a GDA model to training set given by x and y.

    Args:
        x: Training example inputs. Shape (m, n).
        y: Training example labels. Shape (m,).

    Returns:
        phi, mu, sigma: GDA model parameters.
    """
    m, n = x.shape
    idx_to_category, _ = get_category_mappings()
    K = len(idx_to_category)
    
    # Find MLE estimates for each category's phi and mu
    count = np.zeros(K, dtype=np.int32)
    mu = np.zeros((K, n), dtype=np.float32)
    for i in range(m):
        category = int(y[i])
        mu[category] += x[i]
        count[category] += 1
    phi = count/m
    mu /= count[:, np.newaxis]

    # Compute MLE estimate for each category's sigma
    sigma = [np.zeros((n, n), dtype=np.float32) for _ in range(K)]
    for i in range(m):
        category = int(y[i])
        sigma[category] += np.outer(x[i] - mu[category], x[i] - mu[category])
    sigma = [sigma[j]/count[j] for j in range(K)]
    
    return phi, mu, sigma

def get_top3(x_i, phi, mu, sigma, idx_to_category):
    """
    Returns top 3 predicted categories for the given example x_i.
    """
    n = len(x_i)
    K = len(idx_to_category)

    # Compute p(x_i, y=j) for all categories j.
    w = np.zeros(K)
    for j in range(K):
        coeff = 1.0 / np.sqrt(np.linalg.det(sigma[j]))
        a = np.matmul(np.matmul(x_i - mu[j], np.linalg.inv(sigma[j])), x_i - mu[j])
        w[j] = coeff * np.exp(-0.5 * a) * phi[j]

    # Sort likelihoods, return top 3 categories.
    sorted_likelihoods = list(enumerate(w)).sort(key=operator.itemgetter(1))
    return [idx_to_category[idx] for _, idx in sorted_likelihoods[:3]]


def predict(x_val, y_val, phi, mu, sigma):
    """Get model predictions for given dataset."""
    pred = defaultdict(list)
    idx_to_category, _ = get_category_mappings()
    for i, x_i in enumerate(x_val):
        results = get_top3(x_i, phi, mu, sigma, idx_to_category)
        category = idx_to_category[int(y_val[i])]
        pred[category].append(results)
    return pred

if __name__ == "__main__":
    # Load training dataset and train GDA classifier
    x_train, y_train = load_dataset("train")
    print("fitting GDA model...")
    phi, mu, sigma = fit(x_train, y_train)

    # Evaluate classifier on validation set.
    print("computing predictions...")
    x_eval, y_eval = load_dataset("val")
    pred = predict(x_eval, y_eval, phi, mu, sigma)
    compute_scores(pred)
