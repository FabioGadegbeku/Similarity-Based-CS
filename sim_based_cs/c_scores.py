import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel

def laplacian_score(X):
    """Computes the laplacian score of a dataset

    Args:
        X : data set withour labels, column index, or column names

    Returns:
        laplacian_score : laplacian score of the data set
    """
    # Compute the similarity matrix
    S = rbf_kernel(X)
    # Compute the degree matrix
    D = np.diag(S.sum(axis=1))
    # Compute the lapacien matrix
    L = D - S
    # center the data
    X = X - X.mean(axis=0)
    # Compute the lapacien score for each feature
    laplacian_score_num = np.diag(np.dot(X.T, np.dot(L, X)))
    laplacian_score_den = np.diag(np.dot(X.T, np.dot(D, X)))
    laplacian_score = laplacian_score_num / laplacian_score_den
    return laplacian_score
