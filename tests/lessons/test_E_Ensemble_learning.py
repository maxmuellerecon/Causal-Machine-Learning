# Test file for E_Ensemble_learning

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from causalmachinelearning.config import BLD
from causalmachinelearning.lessons.E_Ensemble_learning import (
    make_dataset,
    decision_tree,
    random_forest,
    load_and_split_data,
    standardize_data,
    gradient_boosting_and_accuracy,
    determine_feature_importance,
)
from matplotlib.figure import Figure


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


def test_gradient_boosting_and_accuracy_sample_data_type_gbr():
    """Test the gradient_boosting_and_accuracy function."""
    X_train_std = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([1, 2])
    X_test_std = np.array([[7, 8, 9], [10, 11, 12]])
    y_test = np.array([3, 4])

    gbr, mse, r2 = gradient_boosting_and_accuracy(
        X_train_std, y_train, X_test_std, y_test
    )
    assert isinstance(gbr, GradientBoostingRegressor)


def test_gradient_boosting_and_accuracy_sample_data_type_mse():
    """Test the gradient_boosting_and_accuracy function."""
    X_train_std = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([1, 2])
    X_test_std = np.array([[7, 8, 9], [10, 11, 12]])
    y_test = np.array([3, 4])
    gbr, mse, r2 = gradient_boosting_and_accuracy(
        X_train_std, y_train, X_test_std, y_test
    )
    assert isinstance(mse, float)


def test_gradient_boosting_and_accuracy_sample_data_type_r2():
    """Test the gradient_boosting_and_accuracy function."""
    X_train_std = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([1, 2])
    X_test_std = np.array([[7, 8, 9], [10, 11, 12]])
    y_test = np.array([3, 4])
    gbr, mse, r2 = gradient_boosting_and_accuracy(
        X_train_std, y_train, X_test_std, y_test
    )
    assert isinstance(r2, float)


def test_determine_feature_importance_type():
    """Test the determine_feature_importance function."""
    X_train_std = np.array([[1, 2, 3], [4, 5, 6]])
    X_test_std = np.array([[7, 8, 9], [10, 11, 12]])
    y_train = np.array([1, 3])
    y_test = np.array([1, 2])
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train_std, y_train)
    plt = determine_feature_importance(gbr, X_test_std, y_test)
    assert not isinstance(plt, Figure)
