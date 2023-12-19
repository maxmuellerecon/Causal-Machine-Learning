#Task file for #B_Ridge_and_Lasso

from pathlib import Path
from pytask import task
import pandas as pd
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.B_Ridge_and_lasso import create_data, linear_regression_and_fill_coef_matrix, ridge_regression_and_fill_coef_matrix, lasso_regression_and_fill_coef_matrix



def task_create_data(
    produces={
        "data": BLD / "python" / "Lesson_B" / "data" / "data.csv",
        "plt_data": BLD / "python" / "Lesson_B" / "figures" / "plot_data.png",
    },
):
    """Create data and plot it."""
    data, plt = create_data()
    data.to_csv(produces["data"])
    plt.savefig(produces["plt_data"])


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