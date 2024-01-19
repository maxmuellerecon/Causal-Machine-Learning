# B_Ridge_and_Lasso

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams["figure.figsize"] = 12, 10
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from causalmachinelearning.lessons.__exceptions import (
    _fail_if_not_dataframe,
    _fail_if_not_int,
    _fail_if_alpha_not_float_or_int,
)


###################B.1 Linear Regression##############################################
def create_data():
    """Create data."""
    x = np.array([i * np.pi / 180 for i in range(60, 300, 4)])
    np.random.seed(10)  # Setting seed for reproducibility
    y = np.sin(x) + np.random.normal(0, 0.15, len(x))
    data = pd.DataFrame(np.column_stack([x, y]), columns=["x", "y"])
    for i in range(2, 16):  # power of 1 is already there
        colname = "x_%d" % i  # new var will be x_power
        data[colname] = data["x"] ** i
    return data


def plot_data(data):
    """Plot data."""
    _fail_if_not_dataframe(data)

    plt.plot(data["x"], data["y"], ".")
    return plt


def linear_regression(data, power, models_to_plot):
    """Linear regression and fill coef matrix.

    Args:
        data (DataFrame): Original data
        power (int): Power of the model
        models_to_plot (str): types of models we want to plot

    Returns:
        ret (list): List of coefficients

    """
    _fail_if_not_dataframe(data)
    _fail_if_not_int(power)

    predictors = ["x"]
    if power >= 2:
        predictors.extend(["x_%d" % i for i in range(2, power + 1)])

    # Fit the model
    linreg = LinearRegression()  # Add parentheses here
    linreg.fit(data[predictors], data["y"])
    y_pred = linreg.predict(data[predictors])

    # Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data["x"], y_pred)
        plt.plot(data["x"], data["y"], ".")
        plt.title("Plot for power: %d" % power)

    # Return the result in the pre-defined format
    rss = sum((y_pred - data["y"]) ** 2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret


def linear_regression_and_fill_coef_matrix(data, models_to_plot):
    """WTS: size of coefficients increases exponentially with an increase in model complexity

    Args:
        data (DataFrame): the original data frame
        models_to_plot (str): types of models we want to plot

    Returns:
        coef_matrix (DataFrame): Data of the coefficients, intercepts and rss
    """
    _fail_if_not_dataframe(data)

    col = ["rss", "intercept"] + ["coef_x_%d" % i for i in range(1, 16)]
    ind = ["model_pow_%d" % i for i in range(1, 16)]
    coef_matrix = pd.DataFrame(index=ind, columns=col)
    for i in range(1, 16):
        coef_matrix.iloc[i - 1, 0 : i + 2] = linear_regression(
            data=data, power=i, models_to_plot=models_to_plot
        )
    return coef_matrix


###################B.2 Ridge Regression##############################################
# L2 Regularization: Ridge Regression
# Positive: Ridge regression works well with multicollinearity, includes all of them in model; prevents overfitting
# Negative: No model selection, all predictors are included in the final model
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    """Function for ridge regression."""
    _fail_if_not_dataframe(data)
    _fail_if_alpha_not_float_or_int(alpha)

    # Fit the model
    ridgereg = Ridge(alpha=alpha)
    ridgereg.fit(data[predictors], data["y"])
    y_pred = ridgereg.predict(data[predictors])

    # Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data["x"], y_pred)
        plt.plot(data["x"], data["y"], ".")
        plt.title("Plot for alpha: %.3g" % alpha)

    # Return the result in pre-defined format
    rss = sum((y_pred - data["y"]) ** 2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


def ridge_regression_and_fill_coef_matrix(data):
    """Ridge regression, to see that if value of alpha increases, complexity reduces,
    because we increase bias.

    Args:
        data (DataFrame): Original data

    Returns:
        coef_matrix_ridge (DataFrame): Data of the coefficients, intercepts and rss

    """
    _fail_if_not_dataframe(data)

    # Initialize predictors to be set of 15 powers of x
    predictors = ["x"]
    predictors.extend(["x_%d" % i for i in range(2, 16)])

    # Set the different values of alpha to be tested
    alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

    # Initialize the dataframe for storing coefficients.
    col = ["rss", "intercept"] + ["coef_x_%d" % i for i in range(1, 16)]
    ind = ["alpha_%.2g" % alpha_ridge[i] for i in range(0, 10)]
    coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

    models_to_plot = {1e-15: 231, 1e-10: 232, 1e-4: 233, 1e-3: 234, 1e-2: 235, 5: 236}
    for i in range(10):
        coef_matrix_ridge.iloc[i,] = ridge_regression(
            data, predictors, alpha_ridge[i], models_to_plot
        )
    return coef_matrix_ridge


###################B.3 Lasso Regression##############################################
# L1 Regularization: Lasso Regression
# Positive: Lasso regression can be used to select a subset of predictors
# Negative: Lasso regression (and all regularizations) works poorly when there are high correlations between predictors, selects shrunken coefficients arbitrarily
def lasso_regression(data, predictors, alpha, models_to_plot={}):
    """Function for lasso regression."""
    _fail_if_not_dataframe(data)
    _fail_if_alpha_not_float_or_int(alpha)

    # Fit the model
    lassoreg = Lasso(alpha=alpha, max_iter=100000)
    lassoreg.fit(data[predictors], data["y"])
    y_pred = lassoreg.predict(data[predictors])

    # Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data["x"], y_pred)
        plt.plot(data["x"], data["y"], ".")
        plt.title("Plot for alpha: %.3g" % alpha)

    # Return the result in pre-defined format
    rss = sum((y_pred - data["y"]) ** 2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret


def lasso_regression_and_fill_coef_matrix(data):
    """Lasso regression, to see that if value of alpha increases, complexity reduces,
    because we increase bias.

    Args:
        data (DataFrame): Original data

    Returns:
        coef_matrix_lasso (DataFrame): Data of the coefficients, intercepts and rss (residual sum of squares)

    """
    _fail_if_not_dataframe(data)

    # Initialize predictors to all 15 powers of x
    predictors = ["x"]
    predictors.extend(["x_%d" % i for i in range(2, 16)])

    # Define the alpha values to test
    alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]

    # Initialize the dataframe to store coefficients
    col = ["rss", "intercept"] + ["coef_x_%d" % i for i in range(1, 16)]
    ind = ["alpha_%.2g" % alpha_lasso[i] for i in range(0, 10)]
    coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

    # Define the models to plot
    models_to_plot = {1e-10: 231, 1e-5: 232, 1e-4: 233, 1e-3: 234, 1e-2: 235, 1: 236}

    # Iterate over the 10 alpha values:
    for i in range(10):
        coef_matrix_lasso.iloc[i,] = lasso_regression(
            data, predictors, alpha_lasso[i], models_to_plot
        )
    return coef_matrix_lasso
