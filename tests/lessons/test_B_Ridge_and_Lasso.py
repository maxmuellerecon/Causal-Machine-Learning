# Test file for B_Ridge_and_Lasso

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.B_Ridge_and_lasso import (
    create_data,
    plot_data,
    linear_regression_and_fill_coef_matrix,
    ridge_regression_and_fill_coef_matrix,
    lasso_regression_and_fill_coef_matrix,
    ridge_regression,
    lasso_regression,
)


def test_create_data_shape():
    """Test create_data."""
    data = create_data()
    assert data.shape == (60, 16)


def test_plot_data_non_empty():
    """Test plot_data."""
    data = create_data()
    plot = plot_data(data)
    assert plot is not None


def test_linear_regression_and_fill_coef_matrix_shape():
    """Test linear_regression_and_fill_coef_matrix."""
    data = create_data()
    coef_matrix = linear_regression_and_fill_coef_matrix(
        data, {1: 231, 3: 232, 6: 233, 9: 234, 12: 235, 15: 236}
    )
    assert coef_matrix.shape == (15, 17)


def test_linear_regression_and_fill_coef_matrix_rss():
    """Check if the first column contains 'rss'."""
    data = create_data()
    coef_matrix = linear_regression_and_fill_coef_matrix(
        data, {1: 231, 3: 232, 6: 233, 9: 234, 12: 235, 15: 236}
    )
    assert coef_matrix.columns[0] == "rss"


def test_ridge_regression_fit_type_result():
    """Test ridge_regression function for model fitting."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    models_to_plot = {0.5: 231}
    result = ridge_regression(data, predictors, alpha, models_to_plot)
    assert isinstance(result, list)


def test_ridge_regression_fit_length_result():
    """Test ridge_regression function for model fitting."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    models_to_plot = {0.5: 231}
    result = ridge_regression(data, predictors, alpha, models_to_plot)
    assert len(result) == 3


def test_ridge_regression_fit_type_result_0():
    """Test ridge_regression function for model fitting."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    models_to_plot = {0.5: 231}
    result = ridge_regression(data, predictors, alpha, models_to_plot)
    assert isinstance(result[0], float)


def test_ridge_regression_fit_type_result_1():
    """Test ridge_regression function for model fitting."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    models_to_plot = {0.5: 231}
    result = ridge_regression(data, predictors, alpha, models_to_plot)
    assert isinstance(result[1], float)


def test_ridge_regression_and_fill_coef_matrix_shape():
    """Test ridge_regression_and_fill_coef_matrix."""
    data = create_data()
    coef_matrix_ridge = ridge_regression_and_fill_coef_matrix(data)
    assert coef_matrix_ridge.shape == (10, 17)


def test_ridge_regression_and_fill_coef_matrix_rss():
    """Check if the first column contains 'rss'."""
    data = create_data()
    coef_matrix_ridge = ridge_regression_and_fill_coef_matrix(data)
    assert coef_matrix_ridge.columns[0] == "rss"


def test_lasso_regression_result_type():
    """Test if the result of lasso_regression is a list."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    result = lasso_regression(data, predictors, alpha)
    assert isinstance(result, list)


def test_lasso_regression_result_length():
    """Test if the result of lasso_regression has the correct length."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    result = lasso_regression(data, predictors, alpha)
    assert len(result) == 3


def test_lasso_regression_result_first_element():
    """Test if the first element of the result of lasso_regression is the RSS."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    result = lasso_regression(data, predictors, alpha)
    assert isinstance(result[0], float)


def test_lasso_regression_result_intercept():
    """Test if the second element of the result of lasso_regression is the intercept."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    predictors = ["x"]
    alpha = 0.5
    result = lasso_regression(data, predictors, alpha)
    assert isinstance(result[1], float)


def test_lasso_regression_and_fill_coef_matrix_shape():
    """Test lasso_regression_and_fill_coef_matrix."""
    data = create_data()
    coef_matrix_lasso = lasso_regression_and_fill_coef_matrix(data)
    assert coef_matrix_lasso.shape == (10, 17)


def test_lasso_regression_and_fill_coef_matrix_rss():
    """Check if the first column contains 'rss'."""
    data = create_data()
    coef_matrix_lasso = lasso_regression_and_fill_coef_matrix(data)
    assert coef_matrix_lasso.columns[0] == "rss"
