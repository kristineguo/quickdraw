import numpy as np
from util import *
import operator

from linear_model import LinearModel

def fit(x, y):
    """Fit a GDA model to training set given by x and y.

    Args:
        x: Training example inputs. Shape (m, n).
        y: Training example labels. Shape (m,).

    Returns:
        phi, mu, sigma: GDA model parameters.
    """
    m, n = x.shape
    K = len(idx_to_category)
    
    # Find MLE estimates for each category's phi and mu
    phi = np.zeros(K)
    mu = np.zeros((K, n)) for _ in range(K)
    count = np.zeros(K)
    for i in range(m):
        category = int(y[i])
        mu[category] += x[i]
        phi_count[category] += 1
    phi /= m
    mu /= count[:, np.newaxis]

    # Compute MLE estimate for each category's sigma
    sigma = [np.zeros((n, n)) for _ in range(K)]
    for i in range(m):
        category = int(y[i])
        sigma[category] += np.outer(x[i] - mu[category], x[i] - mu[category])
    sigma = [sigma[j]/count[j] for j in range(K)]

    return phi, mu, sigma

def get_top3(x_i, phi, mu, sigma):
    """
    Returns top 3 predicted categories for the given example x_i.
    """
    n = len(x_i)

    # Compute p(x_i, y=j) for all categories j.
    w = np.zeros(len(mu))
    for j in range(len(mu)):
        coeff = 1.0 / (np.power(2.0*np.pi, n / 2.0) * np.sqrt(np.linalg.det(sigma[j])))
        a = np.matmul(np.matmul((x_i - mu[j]).T, np.linalg.inv(sigma[j])), x_i - mu[j])
        w[j] = coeff * np.exp(-0.5 * a) * phi[j]

    # Sort likelihoods, return top 3 categories.
    sorted_likelihoods = list(enumerate(w)).sort(key=operator.itemgetter(1))
    return [category for _, category in sorted_likelihoods[:3]]


def predict(x, y, phi, mu, sigma):
    """Get model predictions for given dataset."""

    m, n = x.shape
    K = len(idx_to_category)
    idx_to_category, _ = get_category_mappings()
    
    num_correct, top3_correct = defaultdict(int), defaultdict(int)
    for i, x_i in enumerate(x_val):
        results = get_top3(x[i], phi, mu, sigma)
        category = idx_to_category[int(y[i])]
        
        if category == results[0]:
            num_correct[category] += 1
        if category in results:
            top3_correct[category] += 1
    
    print_statistics(num_correct, top3_correct, y)


if __name__ == "__main__":
    # Load training dataset and train GDA classifier
    x_train, y_train = load_dataset("train")
    phi, mu, sigma = fit(x_train, y_train)

    # Evaluate classifier on validation set.
    x_eval, y_eval = load_dataset("val")
    predict(x_eval, y_eval, phi, mu, sigma)
