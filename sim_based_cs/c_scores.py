"""This module contains the functions to preproccess the data and compute the constraint scores and finally plot the results"""
# Explanations of each score can be found in the notebook , more details in the report

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Knn
from tqdm import tqdm
from scipy.stats import rankdata

### ----------------------------------------------------- DATA PROCESSING ------------------------------------------------ ###
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


### ----------------------------------------------------- Unsupervised Scores ------------------------------------------------ ###

# Laplacian Score
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
    # Compute the laplacian score for each feature
    laplacian_score_num = np.diag(np.dot(x.T, np.dot(laplacian_matrix, x)))
    laplacian_score_den = np.diag(np.dot(x.T, np.dot(degree_matrix, x)))
    laplacian_score_matrix = laplacian_score_num / laplacian_score_den

    # Set nan values to infinity
    laplacian_score_matrix[np.isnan(laplacian_score_matrix)] = np.inf
    return laplacian_score_matrix

### ----------------------------------------------------- CONSTRAINT SCORES ------------------------------------------------ ###

# Function to generate constraints
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

            # Set labels other than the top p to NaN
            X[class_indices & ~np.isin(np.arange(len(X)), top_p_indices), -1] = np.nan

    return X

# Function to get the constraints from the dataset
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

# Constraint Score 1
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

    # Set nan values to infinity
    constraint_score_1[np.isnan(constraint_score_1)] = np.inf

    return constraint_score_1


# Constraint Score 2
def constraint_score_2(x,gamma=1):
    """Computes the constraint score 2 of a dataset

    Args:
        x (2D array): data set with available labels
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        The constraint score 2 of the data set
    """

    target = x[:, -1][np.newaxis].T
    x = x[:, :-1]
    must_link, cannot_link = get_constraints(target)

    degree_matrix_must_link = np.diag(must_link.sum(axis=1))
    degree_matrix_cannot_link = np.diag(cannot_link.sum(axis=1))
    laplacian_matrix_must_link = degree_matrix_must_link - must_link
    laplacian_matrix_cannot_link = degree_matrix_cannot_link - cannot_link

    must_link_term = np.diag(np.dot(x.T, np.dot(laplacian_matrix_must_link, x)))
    cannot_link_term = np.diag(np.dot(x.T, np.dot(laplacian_matrix_cannot_link, x)))
    constraint_score_2 = must_link_term - gamma * cannot_link_term

    # Set nan values to infinity
    constraint_score_2[np.isnan(constraint_score_2)] = np.inf

    return constraint_score_2

# Function used to cumpute the SIMILARITY MATRIX KNN used in the constraint score 3
def similarity_matrix_knn(X,gamma,n_neighbors):
    """Computes the similarity matrix of a dataset using the Knn algorithm

    Args:
        X (2D array): numpy array containing the data set with available labels in the last column
        gamma (float): parameter of matrix
        n_neighbors (int): number of neighbors to consider

    Returns:
        2D array : The similarity matrix of the data set
    """
    must_link,_ = get_constraints(X)
    n = len(X)
    similarity_matrix = np.zeros((n,n))
    X = X[:,:-1]
    for i in range(n):
        nearest_neighbors = np.argsort(np.linalg.norm(X - X[i], axis=1))[:n_neighbors]
        for j in range(n):
            if must_link[i,j] == 1:
                similarity_matrix[i,j] = gamma
            elif j in nearest_neighbors:
                similarity_matrix[i,j] = 1
    return similarity_matrix

# Constraint Score 3
def constraint_score_3(X,gamma=100,n_neighbors=5):
    """Computes the constraint score 3 of a dataset

    Args:
        X (_type_): data set with available labels
        gamma (float): parameter of matrix
        n_neighbors (int): number of neighbors to consider
    """
    target = X[:, -1][np.newaxis].T
    X = X[:, :-1]
    _, cannot_link = get_constraints(target)
    similarity_matrix = similarity_matrix_knn(X,gamma,n_neighbors)

    degree_matrix_must_link = np.diag(similarity_matrix.sum(axis=1))
    degree_matrix_cannot_link = np.diag(cannot_link.sum(axis=1))
    laplacian_matrix_must_link = degree_matrix_must_link - similarity_matrix
    laplacian_matrix_cannot_link = degree_matrix_cannot_link - cannot_link

    constraint_score_3_num = np.diag(np.dot(X.T, np.dot(laplacian_matrix_must_link, X)))
    constraint_score_3_den = np.diag(np.dot(X.T, np.dot(laplacian_matrix_cannot_link, X)))
    score_3 = constraint_score_3_num / constraint_score_3_den

    # Set nan values to infinity
    score_3[np.isnan(score_3)] = np.inf

    return score_3

# Constraint Score 4
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
### ----------------------------------------------------- SIMILARITY BASED CONSTRAINT SCORES ------------------------------------------------ ###

# Function to get the nearest prototype
def nearest_prototype(x, prototypes):
    """Given a sample x and a set of prototypes, returns the index of the nearest prototype

    Args:
        x (array): features of a sample
        prototypes (2D array): data set containing the prototypes with their labels

    Returns:
        int : index of the nearest prototype
    """
    distances = np.linalg.norm(x - prototypes, axis=1)
    return np.argmin(distances)

# Fucntion to get the similarity matrix in a semi supervised setting
def similarity_matrix_semi_supervised(X):
    labels = X[:,-1][np.newaxis].T
    mask = ~np.isnan(labels)
    mask = np.squeeze(mask)  # Ensure the boolean array is 1D
    prototypes = X[mask, :]
    labels_prototypes = prototypes[:,-1][np.newaxis].T
    must_link = get_constraints(labels)[0]
    prototypes = prototypes[:,:-1] # we remove the labels from the prototypes
    X = X[:,:-1] # we remove the labels from the data
    nearest_prototypes = np.array([nearest_prototype(x, prototypes) for x in X])
    labels_matrix = labels_prototypes[nearest_prototypes]

    similarity_matrix_semi_supervised = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            if labels_matrix[i] == labels_matrix[j]:
                similarity_matrix_semi_supervised[i,j] = 1
    return similarity_matrix_semi_supervised

# Function used by the semi_supervised_similarity_constraint_score to compute the score of each subset of features
def score_semi_supervised(X):
    """Computes the semi-supervised similarity score of a dataset

    Args:
        X (2D array): numpy array containing the data set with available labels in the last column

    Returns:
        float: The semi-supervised similarity score of the data set
    """
    similarity_matrix = rbf_kernel(X[:,:-1],gamma=1)
    similarity_matrix_ss = similarity_matrix_semi_supervised(X)
    score = np.linalg.norm(similarity_matrix_ss - similarity_matrix)
    return score

# SEMI SUPERVISED SIMILARITY CONSTRAINT SCORE (epsSS)
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

# Function used by the supervised_similarity_constraint_score to compute the score of each subset of features
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

# SUPERVISED SIMILARITY CONSTRAINT SCORE (epsS)
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


 ### ----------------------------------------------------- PLOT YOUR RESULTS ------------------------------------------------ ###

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
    """Plots the accuracy of a Knn classifier for different number of features for the scores in the list

    Args:
        scores (list of functions ): list of scores used to select the features
        train (2D array): numpy array containing the training data set with available labels in the last column
        test (2D array ): numpy array containing the test data set with labels in the last column
        rep (int): number of iterations to compute the accuracy (the final accuracy is the average of the n_iter accuracies)
        p (int): number of labels to keep for each class

    Returns:
        The accuracy of the Knn classifier for different number of features for the scores in the list
    """

    n = np.shape(train)[1]
    All_Accuracy = np.zeros((len(scores),rep,n-1))
    for i in tqdm(range(rep),desc='Rep number',leave=True):
        train_constrained = generate_constraints(train,p)
        for score in scores:
            acc = get_accuracy(score,train,train_constrained,test)
            All_Accuracy[scores.index(score),i] = acc
    return np.mean(All_Accuracy,axis=1)



def auc_score(acc):
    """Computes the area under the curve of the accuracy

    Args:
        acc (numpy array): accuracy of the Knn classifier for different number of features

    Returns:
        float : The area under the curve of the accuracy
    """
    return np.trapz(acc)/len(acc)

 ### ----------------------------------------------------- FEATURE RANKINGS ------------------------------------------------ ###

def rank_matrix(scores,X,rep,p):
    """Computes the rank matrix of a score for a given dataset

    Args:
        scores (function or class instance): list of scores used to select the features
        X (2D array): numpy array containing the data set with available labels in the last column
        rep (int): number of iterations to compute the rank matrix

    Returns:
        3D numpy array : The rank matrix of the scores
    """
    n = np.shape(X)[1]
    X_data = X[:,:-1]
    rank_matrix = np.zeros((len(scores),rep,n-1))
    for i in range(rep):
        X_constraints = generate_constraints(X,p)
        for score in scores:
            if score == laplacian_score :
                rank_matrix[scores.index(score),i] = rankdata(score(X_data))
            elif score == semi_supervised_similarity_constraint_score or score == supervised_similarity_constraint_score:
                rank_matrix[scores.index(score),i] = rankdata(score(X_constraints,n-1))
            else :
                rank_matrix[scores.index(score),i] = rankdata(score(X_constraints))

    return rank_matrix

# Function to compute tau_star in the kendall coefficient
def count_equal_ranks(array):
    """Counts the number of equal ranks in an array

    Args:
        array (1D numpy array): numpy array containing the ranks

    Returns:
        int : The number of equal ranks in the array
    """
    ranks = rankdata(array)

    return len(array) - len(set(ranks))



def kendall_coefficient(R):
    """Computes the Kendall coefficient of a rank matrix

    Args:
        R (2D numpy array): numpy array containing the rank matrix

    Returns:
        float : The Kendall coefficient of the rank matrix
    """
    p = np.shape(R)[0]
    d = np.shape(R)[1]

    # Calcul de Rr
    Rr= np.sum(R,axis=0)

    # Calcul de la moyenne de Rr
    R_bar = np.mean(Rr)

    # Calcul de ∆
    delta = np.sum((Rr - R_bar)**2)

    # Calcul de τ
    tau_v = np.zeros(p)
    for v in range(p):
        tau_v[v] = count_equal_ranks(R[v,:])
    tau = np.sum((tau_v**3 - tau_v))

    # Calcul du coefficient de Kendall
    kendall_coeff = (12 * delta)/(p**2 * (d**3 - d) - p * tau)

    return kendall_coeff

def correct_number_must_link(X,method,p):
    """_summary_

    Args:
    X (2D array): Data set with labels
    method (string): method to deduce new constraints
    p (int): number of labels to keep for each class

    Returns:
    correct_ratio (float): ratio of correct predicted must link
    correct_nml_ratio (float): ratio of correct predicted must link among the true must link
    standard_ratio (float): ratio of standard must link in the true must link (no prediction)
    """

    X_constraints = generate_constraints(X,p)
    all_must_link = get_constraints(X)[0]
    if method == 'nearest prototype':
        similarity_matrix_ss = similarity_matrix_semi_supervised(X_constraints)
        must_link = get_constraints(X_constraints)[0]

        standard_must_link_number = np.sum(must_link)
        true_number_must_link = np.sum(all_must_link)

        correct_number_must_link = 0
        # get the number of true predicted must link
        for i in range(len(X)):
            for j in range(len(X)):
                if similarity_matrix_ss[i,j] == 1 and all_must_link[i,j] == 1:
                    correct_number_must_link += 1
        # get the number of false predicted must link
        false_number_must_link = 0
        for i in range(len(X)):
            for j in range(len(X)):
                if similarity_matrix_ss[i,j] == 1 and all_must_link[i,j] == 0:
                    false_number_must_link += 1

        # get the number of true predicted cannot link
        correct_ratio = correct_number_must_link/(correct_number_must_link+false_number_must_link)
        correct_nml_ratio = correct_number_must_link/true_number_must_link
        standard_ratio = standard_must_link_number/true_number_must_link
        return correct_ratio,correct_nml_ratio,standard_ratio

    elif method == 'knn':
        similarity_matrix_nn = similarity_matrix_knn(X_constraints,1,5)
        must_link = get_constraints(X_constraints)[0]
        standard_must_link_number = np.sum(must_link)

        true_number_must_link = np.sum(all_must_link)
        correct_number_must_link = 0
        # get the number of true predicted must link
        for i in range(len(X)):
            for j in range(len(X)):
                if similarity_matrix_nn[i,j] == 1 and all_must_link[i,j] == 1:
                    correct_number_must_link += 1
        # get the number of false predicted must link
        false_number_must_link = 0
        for i in range(len(X)):
            for j in range(len(X)):
                if similarity_matrix_nn[i,j] == 1 and must_link[i,j] == 0:
                    false_number_must_link += 1
        # get the number of true predicted cannot link
        correct_ratio = correct_number_must_link/(correct_number_must_link+false_number_must_link)
        correct_nml_ratio = correct_number_must_link/true_number_must_link
        standard_ratio = standard_must_link_number/true_number_must_link

        return correct_ratio,correct_nml_ratio,standard_ratio




