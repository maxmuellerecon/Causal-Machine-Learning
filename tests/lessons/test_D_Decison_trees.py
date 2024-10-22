# Test file for D_Decison_trees

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import pickle
import graphviz

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.D_Decision_trees import (
    create_data,
    fit_tree,
    predict_new_value,
    visualize_decision_tree,
    preprocess_data,
)


def test_create_data_type():
    """Test if the output of create_data is a pandas DataFrame."""
    data = create_data()
    assert isinstance(data, pd.DataFrame)


def test_create_data_shape():
    """Test if the output of create_data has the correct shape."""
    data = create_data()
    assert data.shape == (14, 3)


def test_preprocess_data():
    # Create a sample dataset
    dataset = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
    # Call the preprocess_data function
    processed_dataset = preprocess_data(dataset)
    # Check if the output is of type pd.DataFrame
    assert isinstance(processed_dataset, pd.DataFrame)


def test_preprocess_data():
    # Create a sample dataset
    dataset = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
    # Call the preprocess_data function
    processed_dataset = preprocess_data(dataset)
    # Check if the shape of the output is the same as the input
    assert processed_dataset.shape == dataset.shape


def test_fit_tree_is_regressor():
    """Test if the output of fit_tree is a DecisionTreeRegressor."""
    data = create_data()
    regressor = fit_tree(data)
    assert isinstance(regressor, DecisionTreeRegressor)


def test_predict_new_value():
    """Test if the predict_new_value function returns the correct predicted value."""
    # Create a sample DecisionTreeRegressor
    regressor = DecisionTreeRegressor()
    regressor.fit([[1], [2], [3]], [10, 20, 30])
    predicted_value = predict_new_value(regressor, 4)
    assert predicted_value == np.array([30.0])


def test_visualize_decision_tree():
    """Test if the output of visualize_decision_tree is correct."""
    regressor = pickle.load(
        open(BLD / "python" / "Lesson_D" / "model" / "regressor.pkl", "rb")
    )
    # Create sample feature names
    feature_names = ["Production Cost"]
    # Call the visualize_decision_tree function
    graph = visualize_decision_tree(regressor, feature_names)
    # Check if the returned graph is an instance of graphviz.Source
    assert isinstance(graph, graphviz.Source)
