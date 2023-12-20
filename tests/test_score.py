""" This Module contains the tests for the c_score module  """
import pytest
import numpy as np
import os
import sys
file_dir = os.path.dirname('Similarity-Based-CS')
sys.path.append(file_dir)
from sim_based_cs.c_scores import *

def test_basic():
    assert 1 == 1

def test_laplacian_score():
    """ This function tests the laplacian_score function """
    # Generate a random dataset
    x = np.random.rand(100, 10)
    # Compute the laplacian score
    laplacian_score_matrix = laplacian_score(x)
    # Check if the laplacian score matrix is a numpy array
    assert isinstance(laplacian_score_matrix, np.ndarray)
    # Check if the laplacian score matrix is of the right shape
    assert laplacian_score_matrix.shape == (10,)

def test_get_constraints():
    """ This function tests the get_constraints function """
    x = np.full((5, 5), 1)
    # Compute the constraints
    must_link, cannot_link = get_constraints(x)
    # Check if the results are the right shape
    assert must_link.shape == (5, 5)
    assert cannot_link.shape == (5, 5)
    # check if we get the correct results
    assert np.all(must_link == np.full((5, 5), 1))
    assert np.all(cannot_link == np.full((5, 5), 0))


def test_generate_constraints():
    """ This function tests the generate_constraints function """
    # Generate a random dataset
    x = np.random.rand(100, 10)
    # Generate 10 constraints
    target = generate_constraints(x, 10)
    # Check if the result is a numpy array
    assert isinstance(target, np.ndarray)
    # Check if the result is of the right shape
    assert target.shape == (100,)
    # Check if the result contains the right number of nan values
    assert np.sum(np.isnan(target)) == 90




