"""This module contains the functions to compute the constraint scores of a dataset"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def laplacian_score(x):
    """Computes the laplacian score of a dataset

    Args:
        X : data set without labels

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


def get_constraints(x):
    """ gets the constraints from the dataset

    Args:
        x : data set with available labels
    Returns:
        must_link (Must Link Matrix): i,j = 1 if i and j are must link, 0 otherwise
        cannot_link (Cannot Link Matrix): i,j = 1 if i and j are cannot link, 0 otherwise
    """
    classe = x[:, -1]
    must_link = np.zeros((len(classe), len(classe)))
    cannot_link = np.zeros((len(classe), len(classe)))
    for i in range(len(classe)):
        for j in range(len(classe)):
            if classe[i] == classe[j] :
                must_link[i, j] = 1
            if classe[i] != classe[j] and np.isnan(classe[i]) == False and np.isnan(classe[j]) == False:
                cannot_link[i, j] = 1
    return must_link, cannot_link


## function that generates p constraints by setting n-p targets to NaN randomly and returns the target with n-p nan values
def generate_constraints(x, p):
    """ generates p constraints by setting n-p targets to NaN randomly and returns the target with n-p nan values
    Args : x : data set with available labels
              p : number of constraints
    Returns : target : target with n-p nan values
    """
    target = x[:, -1]

    # Randomly select n-p indices to set to nan
    nan_indices = np.random.choice(target.size, target.size - p, replace=False)

    # Set selected indices to nan
    target = target.astype(np.float64)
    target[nan_indices] = np.nan

    return target


def constraint_score_1(x,target):
    """Computes the constraint score 1 of a dataset

    Args:
        x : data set without labels
        target : available labels

    Returns:
        constraint_score_1 : constraint score 1 of the data set
    """
    # Get the constraints
    must_link, cannot_link = get_constraints(target)

    degree_matrix_must_link = np.diag(must_link.sum(axis=1))
    degree_matrix_cannot_link = np.diag(cannot_link.sum(axis=1))
    laplacian_matrix_must_link = degree_matrix_must_link - must_link
    laplacian_matrix_cannot_link = degree_matrix_cannot_link - cannot_link

    constraint_score_1_num = np.diag(np.dot(x.T, np.dot(laplacian_matrix_must_link, x)))
    constraint_score_1_den = np.diag(np.dot(x.T, np.dot(laplacian_matrix_cannot_link, x)))
    constraint_score_1 = constraint_score_1_num / constraint_score_1_den
    return constraint_score_1