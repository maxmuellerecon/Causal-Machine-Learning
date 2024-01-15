#Test file for E_Ensemble_learning

from sklearn.tree import DecisionTreeRegressor  
import numpy as np
import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from causalmachinelearning.config import BLD
from causalmachinelearning.data_management.E_Ensemble_learning import make_dataset, decision_tree, random_forest, load_and_split_data, standardize_data


def test_make_dataset_type_X():
    """Test the make_dataset function."""
    X, y, df, coef = make_dataset()
    assert isinstance(X, np.ndarray)
    
def test_make_dataset_type_y():
    """Test the make_dataset function."""
    X, y, df, coef = make_dataset()
    assert isinstance(y, np.ndarray)

def test_make_dataset_type_df():
    """Test the make_dataset function."""
    X, y, df, coef = make_dataset()
    assert isinstance(df, pd.DataFrame)

def test_make_dataset_type_coef():
    """Test the make_dataset function."""
    X, y, df, coef = make_dataset()
    assert isinstance(coef, np.ndarray)
    
def test_decision_tree_model():
    """Test the decision_tree function."""
    X = np.load(BLD / "python" / "Lesson_E" / "data" / "X.npy")
    y = np.load(BLD / "python" / "Lesson_E" / "data" / "y.npy")
    tree_model, r2 = decision_tree(X, y)
    assert isinstance(tree_model, DecisionTreeRegressor)
    
def test_decision_tree_r2():
    """Test the decision_tree function."""
    X = np.load(BLD / "python" / "Lesson_E" / "data" / "X.npy")
    y = np.load(BLD / "python" / "Lesson_E" / "data" / "y.npy")
    tree_model, r2 = decision_tree(X, y)
    assert isinstance(r2, float)
    
def test_random_forest_model():
    """Test the random_forest function."""
    X = np.load(BLD / "python" / "Lesson_E" / "data" / "X.npy")
    y = np.load(BLD / "python" / "Lesson_E" / "data" / "y.npy")
    forest_model, r2 = random_forest(X, y)
    assert isinstance(forest_model, RandomForestRegressor)
    
def test_random_forest_r2():
    """Test the random_forest function."""
    X = np.load(BLD / "python" / "Lesson_E" / "data" / "X.npy")
    y = np.load(BLD / "python" / "Lesson_E" / "data" / "y.npy")
    forest_model, r2 = random_forest(X, y)
    assert isinstance(r2, float)
    
def test_load_and_split_data_type_X_train():
    """Test the load_and_split_data function."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert isinstance(X_train, np.ndarray)

def test_load_and_split_data_type_X_test():
    """Test the load_and_split_data function."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert isinstance(X_test, np.ndarray)
    
def test_load_and_split_data_type_y_train():
    """Test the load_and_split_data function."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert isinstance(y_train, np.ndarray)

def test_load_and_split_data_type_y_test():
    """Test the load_and_split_data function."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert isinstance(y_test, np.ndarray)

def test_standardize_data_X_train_std():
    """Test the standardize_data function."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    assert isinstance(X_train_scaled, np.ndarray)
    
def test_standardize_data_X_test_std():
    """Test the standardize_data function."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    assert isinstance(X_test_scaled, np.ndarray)
    
