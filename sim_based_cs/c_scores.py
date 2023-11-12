"""This module contains the functions to compute the constraint scores of a dataset"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def laplacian_score(x):
    """Computes the laplacian score of a dataset

    Args:
        X : data set withour labels, column index, or column names

    Returns:
        laplacian_score : laplacian score of the data set
    """
    # Compute the similarity matrix
    simalarity_matrix = rbf_kernel(x)
    # Compute the degree matrix
    degree_matrix = np.diag(simalarity_matrix.sum(axis=1))
    # Compute the lapacien matrix
    laplacian_matrix = degree_matrix - simalarity_matrix
    # center the data
    x = x - x.mean(axis=0)
    # Compute the lapacien score for each feature
    laplacian_score_num = np.diag(np.dot(x.T, np.dot(laplacian_matrix, x)))
    laplacian_score_den = np.diag(np.dot(x.T, np.dot(degree_matrix, x)))
    laplacian_score_matrix = laplacian_score_num / laplacian_score_den
    return laplacian_score_matrix
