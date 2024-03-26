# Similarity-Based-CS
This project addresses the curse of dimensionality in feature selection methods, with a focus on supervised and semi-supervised learning. It introduces a novel constraint score that evaluates the relevance of feature subsets in a lower-dimensional subspace, taking into account correlations between features and avoiding high-dimensional original feature spaces.

## Key Components :

- Curse of Dimensionality: The project tackles the challenges associated with high-dimensional data and the impact on machine learning tasks.

- Feature Selection: Describes the process of selecting a subset of relevant features to improve model effectiveness.

- Constraint Scores: Explains the use of must-link and cannot-link constraints and their limitations when evaluating individual features.

- Correlations Between Features: Highlights the importance of considering feature interactions and dependencies in real-world data.

- New Constraint Score: A novel constraint score that operates on feature subsets in a lower-dimensional subspace.

## How To Use :

- Information and descriptions of the state of the art constraint scores and our new similarity based constraint score can be found in the report pdf.

- To use the implementations of these scores clone the repository and the scores can be found in the c_scores.py file inside the sim_based_cs folder.

- In c_scores.py you'll also find functions for preprocessing your data, plotting results, and computing the rank matrix.
  
- In the notebook you can find the results obtained by my implementations.

If there's any problem or questions don't hesitate to contact me !

## Code Quality :
![Alt text](./coverage.svg)
![pylint](https://img.shields.io/badge/pylint-10.00-brightgreen?logo=python&logoColor=white)
