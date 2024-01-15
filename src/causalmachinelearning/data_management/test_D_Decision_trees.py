import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from causalmachinelearning.data_management.D_Decision_trees import fit_tree, preprocess_data

@pytest.fixture
def dataset():
    # Create a sample dataset for testing
    dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return dataset

def test_fit_tree(dataset):
    # Preprocess the dataset
    preprocessed_dataset = preprocess_data(dataset)
    
    # Extract X and y from the preprocessed dataset
    X = preprocessed_dataset[:, 1].reshape(-1, 1)
    y = preprocessed_dataset[:, 2]
    
    # Fit the tree
    regressor = fit_tree(dataset)
    
    # Check if the regressor is an instance of DecisionTreeRegressor
    assert isinstance(regressor, DecisionTreeRegressor)
    
    # Check if the regressor is fitted with the correct X and y data
    assert np.array_equal(regressor.predict(X), y)