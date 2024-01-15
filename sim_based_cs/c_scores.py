"""This module contains the functions to compute the constraint scores of a dataset"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
def laplacian_score(x):
    """Computes the laplacian score of a dataset

    Args:
        X : data set without labels

    Returns:
        laplacian_score : laplacian score of the data set
    """
    # Center and normalize the data
    scaler = StandardScaler()
    scaler.fit_transform(x)
    # Compute the similarity matrix
    simalarity_matrix = rbf_kernel(x)
    # Compute the degree matrix
    degree_matrix = np.diag(simalarity_matrix.sum(axis=1))
    # Compute the lapacien matrix
    laplacian_matrix = degree_matrix - simalarity_matrix
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


## function that makes only p labels available for each class
def generate_constraints(X, p):
    """Sets only p labels available for each class

    Args:
        X (2D numpy array ): numpy array containing the data set with available labels in the last column
        p (int): number of labels to keep for each class

    Returns:
        X : numpy array containing the data set with only p labels available for each class
    """

    #Get target column from X
    target_column = X[:,-1]

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(target_column, return_counts=True)

    # Iterate over each class and set labels other than the top p to NaN
    for class_label, count in zip(unique_classes, class_counts):
        class_indices = target_column == class_label

        if count > p:
            top_p_indices = np.random.choice(np.where(class_indices)[0], size=p, replace=False)
            # Use np.random.choice instead of sample for a NumPy array

            # Set labels other than the top p to NaN
            X[class_indices & ~np.isin(np.arange(len(X)), top_p_indices), -1] = np.nan

    return X

def constraint_score_1(x):
    """Computes the constraint score 1 of a dataset

    Args:
        x : data set with available labels

    Returns:
        constraint_score_1 : constraint score 1 of the data set
    """
    # Get the constraints
    target = x[:, -1][np.newaxis].T
    x = x[:, :-1]
    must_link, cannot_link = get_constraints(target)

    degree_matrix_must_link = np.diag(must_link.sum(axis=1))
    degree_matrix_cannot_link = np.diag(cannot_link.sum(axis=1))
    laplacian_matrix_must_link = degree_matrix_must_link - must_link
    laplacian_matrix_cannot_link = degree_matrix_cannot_link - cannot_link

    constraint_score_1_num = np.diag(np.dot(x.T, np.dot(laplacian_matrix_must_link, x)))
    constraint_score_1_den = np.diag(np.dot(x.T, np.dot(laplacian_matrix_cannot_link, x)))
    constraint_score_1 = constraint_score_1_num / constraint_score_1_den
    return constraint_score_1


def similarity_matrix_score(x,m):
    """Computes the similarity matrix score of a dataset keeping the m first features

    Args:
        x : data set with available labels
        m : number of features to keep

    Returns:
        The similarity matrix score for each feature
    """
    target = x[:, -1]
    x = x[:, :-1]
    # Center and normalize the data
    scaler = StandardScaler()
    scaler.fit_transform(x)
    # Compute the similarity matrix for each feature
    similarity_matrix_score =[]
    #get the constraints
    must_link, cannot_link = get_constraints(target)
    for i in range(m+1,x.shape[1]-1):
        similarity_matrix = (rbf_kernel(x[:,m:i].reshape(-1, 1)))
        similarity_matrix_supervised = np.zeros((len(target), len(target)))
        for j in range(len(target)):
            for k in range(len(target)):
                if must_link[j, k] == 1:
                    similarity_matrix_supervised[j, k] = 1
                if cannot_link[j, k] == 1:
                    similarity_matrix_supervised[j, k] = 0
                else:
                    similarity_matrix_supervised[j, k] = similarity_matrix[j, k]
        similarity_matrix_score.append(np.linalg.norm(similarity_matrix_supervised - similarity_matrix))

    return similarity_matrix_score

# def score_supervised(X):
#     target = X[:,-1][np.newaxis].T
#     X = X[:,:-1]
#     must_link, cannot_link = get_constraints(target)
#     similarity_matrix = rbf_kernel(X)
#     similarity_matrix_supervised = np.zeros((len(target), len(target)))
#     for j in range(len(target)):
#         for k in range(len(target)):
#             if must_link[j, k] == 1:
#                 similarity_matrix_supervised[j, k] = 1
#             if cannot_link[j, k] == 1:
#                 similarity_matrix_supervised[j, k] = 0
#             else:
#                 similarity_matrix_supervised[j, k] = similarity_matrix[j, k]
#     score = np.linalg.norm(similarity_matrix_supervised - similarity_matrix)
#     return score

def similarity_constraint_score(X,score,m):
    """Creates a subset of features of size m that maximizes the similarity matrix score (Greedy Algorithm)

    Args:
        X (2D array): numpy array containing the data set with available labels in the last column
        score (function): function to compute the similarity score of a data set supervied or semi-supervised
        m ( int ): number of features to keep

    Returns:
        int : The best subset of size m of features
    """
    target = X[:, -1][np.newaxis].T
    X = X[:, :-1]
    n = X.shape[1]
    # Center and normalize the data
    scaler = StandardScaler()
    scaler.fit_transform(X)
    selected_features = []
    for j in range(m):
        similarity_matrix_score =[]
        if j == 0:
            for i in range(n):
                similarity_matrix_score.append(score(np.concatenate((X[:,i][np.newaxis].T,target), axis=1)))
            feature_rank = np.argsort(similarity_matrix_score)
            selected_features.append(feature_rank[0])
            features = np.delete(X, feature_rank[1:], axis=1)
        else :
            for i in range(n):
                if i not in selected_features:
                    fi = np.concatenate((features,X[:,i][np.newaxis].T), axis=1)
                    fi = np.concatenate((fi,target), axis=1)
                    similarity_matrix_score.append(score(fi))
                else:
                    similarity_matrix_score.append(np.inf)
            feature_rank = np.argsort(similarity_matrix_score)
            selected_features.append(feature_rank[0])
            features = np.concatenate((features,X[:,feature_rank[0]][np.newaxis].T), axis=1)
    return selected_features

def score_supervised(X):
    """Computes the supervised similarity score of a dataset

    Args:
        X (2D array): numpy array containing the data set with available labels in the last column

    Returns:
       float : The supervised similarity score of the data set
    """
    labels = X[:,-1][np.newaxis].T
    X = X[:,:-1]
    must_link, cannot_link = get_constraints(labels)
    similarity_matrix = rbf_kernel(X)
    similarity_matrix_supervised = np.zeros((len(labels), len(labels)))
    for j in range(len(labels)):
        for k in range(len(labels)):
            if must_link[j, k] == 1:
                similarity_matrix_supervised[j, k] = 1
            if cannot_link[j, k] == 1:
                similarity_matrix_supervised[j, k] = 0
            else:
                similarity_matrix_supervised[j, k] = similarity_matrix[j, k]
    score = np.linalg.norm(similarity_matrix_supervised - similarity_matrix)
    return score

def nearest_prototype(x, prototypes):
    """Given a sample x and a set of prototypes, returns the index of the nearest prototype

    Args:
        x (array): features of a sample
        prototypes (2D array): data set containing the prototypes with their labels

    Returns:
        int : index of the nearest prototype
    """
    distance = np.zeros(len(prototypes))
    for i in range(len(prototypes)):
        distance[i] = np.linalg.norm(x - prototypes[i])
    return np.argmin(distance)

def score_semi_supervised(X):
    """Computes the semi-supervised similarity score of a dataset

    Args:
        X (2D array): numpy array containing the data set with available labels in the last column

    Returns:
        float: The semi-supervised similarity score of the data set
    """
    labels = X[:,-1][np.newaxis].T
    mask = ~np.isnan(labels)
    mask = np.squeeze(mask)  # Ensure the boolean array is 1D
    prototypes = X[mask, :]
    labels_prototypes = prototypes[:,-1][np.newaxis].T
    must_link = get_constraints(labels)[0]
    prototypes = prototypes[:,:-1] # we remove the labels from the prototypes
    X = X[:,:-1] # we remove the labels from the data
    similarity_matrix = rbf_kernel(X) #The true similarity matrix
    similarity_matrix_semi_supervised = np.zeros((len(X), len(X))) #The constructed similarity matrix with constraints
    for i in range(len(X)):
        NP_Xi = nearest_prototype(X[i], prototypes)
        for j in range(len(X)):
            NP_Xj = nearest_prototype(X[j], prototypes)
            if labels_prototypes[NP_Xi] == labels_prototypes[NP_Xj] or must_link[i, j] == 1:
                similarity_matrix_semi_supervised[i, j] = 1
            else:
                similarity_matrix_semi_supervised[i, j] = 0
    score = np.linalg.norm(similarity_matrix_semi_supervised - similarity_matrix)
    return score
