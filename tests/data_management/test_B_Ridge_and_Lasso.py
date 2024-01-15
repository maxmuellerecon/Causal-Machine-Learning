#Test file for B_Ridge_and_Lasso

import pandas as pd
import pytest
pytestmark = pytest.mark.filterwarnings("ignore")

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.B_Ridge_and_lasso import create_data, plot_data, linear_regression_and_fill_coef_matrix, ridge_regression_and_fill_coef_matrix, lasso_regression_and_fill_coef_matrix

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
    coef_matrix = linear_regression_and_fill_coef_matrix(data, {1: 231, 3: 232, 6: 233, 9: 234, 12: 235, 15: 236})
    assert coef_matrix.shape == (15, 17)
    
def test_linear_regression_and_fill_coef_matrix_rss():
    """Check if the first column contains 'rss'"""
    data = create_data()
    coef_matrix = linear_regression_and_fill_coef_matrix(data, {1: 231, 3: 232, 6: 233, 9: 234, 12: 235, 15: 236})
    assert coef_matrix.columns[0] == 'rss'
    
def test_ridge_regression_and_fill_coef_matrix_shape():
    """Test ridge_regression_and_fill_coef_matrix."""
    data = create_data()
    coef_matrix_ridge = ridge_regression_and_fill_coef_matrix(data)
    assert coef_matrix_ridge.shape == (10, 17)

def test_ridge_regression_and_fill_coef_matrix_rss():
    """Check if the first column contains 'rss'"""
    data = create_data()
    coef_matrix_ridge = ridge_regression_and_fill_coef_matrix(data)
    assert coef_matrix_ridge.columns[0] == 'rss'
    
def test_lasso_regression_and_fill_coef_matrix_shape():
    """Test lasso_regression_and_fill_coef_matrix."""
    data = create_data()
    coef_matrix_lasso = lasso_regression_and_fill_coef_matrix(data)
    assert coef_matrix_lasso.shape == (10, 17)
    
def test_lasso_regression_and_fill_coef_matrix_rss():
    """Check if the first column contains 'rss'"""
    data = create_data()
    coef_matrix_lasso = lasso_regression_and_fill_coef_matrix(data)
    assert coef_matrix_lasso.columns[0] == 'rss'