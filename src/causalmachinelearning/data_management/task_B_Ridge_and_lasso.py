#Task file for #B_Ridge_and_Lasso

import pandas as pd

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.B_Ridge_and_lasso import create_data, plot_data, linear_regression_and_fill_coef_matrix, ridge_regression_and_fill_coef_matrix, lasso_regression_and_fill_coef_matrix


def task_create_data(
    produces={
        "data": BLD / "python" / "Lesson_B" / "data" / "data.csv",
    },
):
    """Create data"""
    data = create_data()
    data.to_csv(produces["data"])
    
    
def task_plot_data(
    depends_on=BLD / "python" / "Lesson_B" / "data" / "data.csv",
    produces={
        "plot": BLD / "python" / "Lesson_B" / "figures" / "plot.png",
    },
    ):
    """Plot data."""
    data = pd.read_csv(depends_on)
    plot = plot_data(data)
    plot.savefig(produces["plot"])


def task_linear_regression_and_fill_coef_matrix(
    depends_on=BLD / "python" / "Lesson_B" / "data" / "data.csv",
    produces=BLD / "python" / "Lesson_B" / "data" / "coef_matrix_linreg.csv",
):
    """Linear regression and fill coef matrix."""
    data = pd.read_csv(depends_on)
    coef_matrix = linear_regression_and_fill_coef_matrix(data, {1: 231, 3: 232, 6: 233, 9: 234, 12: 235, 15: 236})
    coef_matrix.to_csv(produces)
    

def task_ridge_regression_and_fill_coef_matrix(
    depends_on=BLD / "python" / "Lesson_B" / "data" / "data.csv",
    produces=BLD / "python" / "Lesson_B" / "data" / "coef_matrix_ridge.csv",
):
    """Ridge regression and fill coef matrix."""
    data = pd.read_csv(depends_on)
    coef_matrix_ridge = ridge_regression_and_fill_coef_matrix(data)
    coef_matrix_ridge.to_csv(produces)
    
    
def task_lasso_regression_and_fill_coef_matrix(
    depends_on=BLD / "python" / "Lesson_B" / "data" / "data.csv",
    produces=BLD / "python" / "Lesson_B" / "data" / "coef_matrix_lasso.csv",
):
    """Lasso regression and fill coef matrix."""
    data = pd.read_csv(depends_on)
    coef_matrix_lasso = lasso_regression_and_fill_coef_matrix(data)
    coef_matrix_lasso.to_csv(produces)