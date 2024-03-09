"""This module contains the functions to compute the constraint scores of a dataset"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Knn
def laplacian_score(x):
    """Computes the laplacian score of a dataset

    Args:
        X : data set without labels

    Returns:
        laplacian_score : laplacian score of the data set
    """
    # Compute the similarity matrix
    simalarity_matrix = rbf_kernel(x,gamma=1)
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
    X = X.copy()
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

def constraint_score_4(x):
    """

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_data = x[:,:-1]
    ls = laplacian_score(x_data)
    cs1 = constraint_score_1(x)

    return ls*cs1

def semi_supervised_similarity_constraint_score(X,m):
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
    selected_features = []
    for j in range(m):
        similarity_matrix_score =[]
        if j == 0:
            for i in range(n):
                similarity_matrix_score.append(score_semi_supervised(np.concatenate((X[:,i][np.newaxis].T,target), axis=1)))
            feature_rank = np.argsort(similarity_matrix_score)
            selected_features.append(feature_rank[0])
            features = np.delete(X, feature_rank[1:], axis=1)
        else :
            for i in range(n):
                if i not in selected_features:
                    fi = np.concatenate((features,X[:,i][np.newaxis].T), axis=1)
                    fi = np.concatenate((fi,target), axis=1)
                    similarity_matrix_score.append(score_semi_supervised(fi))
                else:
                    similarity_matrix_score.append(np.inf)
            feature_rank = np.argsort(similarity_matrix_score)
            selected_features.append(feature_rank[0])
            features = np.concatenate((features,X[:,feature_rank[0]][np.newaxis].T), axis=1)
    return selected_features

def supervised_similarity_constraint_score(X,m):
    """Creates a subset of features of size m that maximizes the similarity matrix score (Greedy Algorithm)

    Args:
        X (2D array): numpy array containing the data set with available labels in the last column
        m ( int ): number of features to keep

    Returns:
        int : The best subset of size m of features
    """
    target = X[:, -1][np.newaxis].T
    X = X[:, :-1]
    n = X.shape[1]
    selected_features = []
    for j in range(m):
        similarity_matrix_score =[]
        if j == 0:
            for i in range(n):
                similarity_matrix_score.append(score_supervised(np.concatenate((X[:,i][np.newaxis].T,target), axis=1)))
            feature_rank = np.argsort(similarity_matrix_score)
            selected_features.append(feature_rank[0])
            features = np.delete(X, feature_rank[1:], axis=1)
        else :
            for i in range(n):
                if i not in selected_features:
                    fi = np.concatenate((features,X[:,i][np.newaxis].T), axis=1)
                    fi = np.concatenate((fi,target), axis=1)
                    similarity_matrix_score.append(score_supervised(fi))
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
    similarity_matrix = rbf_kernel(X,gamma=1)
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
    similarity_matrix = rbf_kernel(X,gamma= 1) #The true similarity matrix
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


def get_accuracy(score,train,train_constrained,test):
    """Plots the accuracy of a Knn classifier for different number of features for a given score

    Args:
        score (function or class instance): score used to select the features
        n_iter (int): number of iterations to compute the accuracy (the final accuracy is the average of the n_iter accuracies)
        train (2D array): numpy array containing the training data set with available labels in the last column
        test (2D array ): numpy array containing the test data set with labels in the last column
        p (int): number of labels to keep for each class

    Returns:
        numpy array : The accuracy of the Knn classifier for different number of features
    """
    n = np.shape(train)[1]
    knn = Knn(n_neighbors=1)
    X_train = train[:,:-1]
    y_train = train[:,-1]
    X_test = test[:,:-1]
    y_test = test[:,-1]

    if score == laplacian_score :
        features = np.argsort(score(X_train))

    elif score == semi_supervised_similarity_constraint_score or score == supervised_similarity_constraint_score:
        features = np.argsort(score(train_constrained,n-1))

    else :
        features = np.argsort(score(train_constrained))

    Accuracy = np.zeros(n-1)

    for i in range(1,n):
        # i features to keep
        X_train_fs = np.delete(X_train, features[i:], axis=1)
        X_test_fs = np.delete(X_test, features[i:], axis=1)

        #scale the data
        scaler = StandardScaler()
        X_train_fs = scaler.fit_transform(X_train_fs)
        X_test_fs = scaler.transform(X_test_fs)

        # Fit the Knn classifier
        knn.fit(X_train_fs, y_train)
        y_pred = knn.predict(X_test_fs)

        # Compute the accuracy
        Accuracy[i-1] = np.mean(y_pred == y_test)

    return Accuracy

def plot_accuracy(scores,train,test,rep,p):

    n = np.shape(train)[1]
    All_Accuracy = np.zeros((len(scores),rep,n-1))
    for i in range(rep):
        train_constrained = generate_constraints(train,p)
        for score in scores:
            acc = get_accuracy(score,train,train_constrained,test)
            All_Accuracy[scores.index(score),i] = acc
    return np.mean(All_Accuracy,axis=1)



def auc_score(acc):
    return np.trapz(acc)/len(acc)

def split_dataset(X):
    """Splits the dataset into a training and test set with half of the labels available for each class

    Args:
        X (2D numpy array): numpy array containing the data set with labels in the last column

    Returns:
        (2D array, 2D array) : The training and test sets
    """
    # Get the unique classes from the last column
    classes = np.unique(X[:, -1])

    # Initialize lists to store indices of each class
    class_indices = {cls: [] for cls in classes}

    # Group indices of each class
    for idx, cls in enumerate(X[:, -1]):
        class_indices[cls].append(idx)

    # Initialize lists to store indices for training and test sets
    train_indices = []
    test_indices = []

    # Split each class into training and test sets
    for cls in classes:
        # Shuffle indices of the current class
        np.random.shuffle(class_indices[cls])

        # Calculate the split index (half of the class size)
        split_idx = len(class_indices[cls]) // 2

        # Add half of the indices to the training set
        train_indices.extend(class_indices[cls][:split_idx])

        # Add the other half of the indices to the test set
        test_indices.extend(class_indices[cls][split_idx:])

    # Create training and test sets
    X_train = X[train_indices]
    X_test = X[test_indices]

    return X_train, X_test

def rank_matrix(scores,X,rep,p):
    """Computes the rank matrix of a score for a given dataset

    Args:
        scores (function or class instance): list of scores used to select the features
        X (2D array): numpy array containing the data set with available labels in the last column
        rep (int): number of iterations to compute the rank matrix

    Returns:
        3 numpy array : The rank matrix of the scores
    """
    n = np.shape(X)[1]
    X_data = X[:,:-1]
    rank_matrix = np.zeros((len(scores),rep,n-1))
    for i in range(rep):
        X_constraints = generate_constraints(X,p)
        for score in scores:
            if score == laplacian_score :
                rank_matrix[scores.index(score),i] = np.argsort(score(X_data))
            elif score == semi_supervised_similarity_constraint_score or score == supervised_similarity_constraint_score:
                rank_matrix[scores.index(score),i] = np.argsort(score(X_constraints,n-1))
            else :
                rank_matrix[scores.index(score),i] = np.argsort(score(X_constraints))
    return rank_matrix



