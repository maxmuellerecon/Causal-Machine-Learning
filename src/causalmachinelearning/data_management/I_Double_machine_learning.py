# I_Double-machine learning

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict


# Positives: Works for continuous and binary treatments and has rigorous validity
#####################I.1 Recap: Frisch-Waugh-Lovell Theorem########################
def plot_pattern(df, x_var, y_var):
    """Plot pattern of data"""
    np.random.seed(123)
    plot = sns.scatterplot(data=df.sample(1000), x=x_var, y=y_var, hue="weekday")
    return plot


def fwl_theorem(train):
    """Apply the Frisch-Waugh-Lovell Theorem

    Args:
        df (DataFrame): The training data

    Returns:
        table1 (DataFrame): The regression table
    """
    # FWL Theorem
    # 1.) Regress Outcome (Sales) on Covariates (Weekday, Temp, Cost)
    my = smf.ols("sales~temp+C(weekday)+cost", data=train).fit()
    # 2.) Regress Treatment (Price) on Covariates (Weekday, Temp, Cost)
    mt = smf.ols("price~temp+C(weekday)+cost", data=train).fit()
    # 3.) Obtain the residuals from the 2 regressions and regress y - y_hat on t - t_hat
    table1 = (
        smf.ols(
            "sales_res~price_res",
            data=train.assign(
                sales_res=my.resid, price_res=mt.resid  # sales residuals
            ),  # price residuals
        )
        .fit()
        .summary()
    )
    return table1


def verify_fwl_theorem(train):
    """Verify that FWL is the same as usual regression: it is exactly the same (-4.004)"""
    table2 = smf.ols("sales~price+temp+C(weekday)+cost", data=train).fit().summary()
    return table2


# #####################I.2 Parametric Double ML ATE############################################
# #Key Idea: use ML model to construct the residual functions, so no OLS, hence capture non-linearities, interactions
# #between X and T and X and Y
# #1.) Regress Outcome (Sales) on Covariates (Weekday, Temp, Cost) with flexible ML model
# #2.) Regress Treatment (Price) on Covariates (Weekday, Temp, Cost) with flexible ML model
# #3.) Obtain the residuals from the 2 regressions
# #4.) Regress y - my_hat on t - mt_hat, get causal effect, estimate by e.g. OLS
# #Positive: Gain flexibility
# #Negative: Could overfit in steps 1 and 2
# #Solution to overfitting: Use cross-fitting, where we fit model on one part of data and make residuals and predictions in other


# 1.) Debias the treatment +3.) obtain residuals for t
def debias_treatment(train):
    """Debias the treatment"""
    # Define the ML model
    T = "price"
    X = ["temp", "weekday", "cost"]
    debias_m = LGBMRegressor(max_depth=3)
    # Estimate residuals: t_tilde = t - t_hat(from ML model)
    # add train[T].mean for visualization
    train_pred = train.assign(
        price_res=train[T]
        - cross_val_predict(debias_m, train[X], train[T], cv=5)
        + train[T].mean()
    )
    return train_pred


def plot_debiased(df):
    """Visualize debiased treatment"""
    np.random.seed(123)
    sns.scatterplot(data=df.sample(1000), x="price_res", y="sales", hue="weekday")
    return plt


# 2.) Reduce the variance from y +3.) obtain residuals for Y
def denoise_outcome(train, train_pred):
    """Denoise the outcome"""
    Y = "sales"
    X = ["temp", "weekday", "cost"]
    denoise_m = LGBMRegressor(max_depth=3)
    # Estimate residuals: y_tilde = y - y_hat(from ML model)
    # add train[Y].mean for visualization
    train_pred_y = train_pred.assign(
        sales_res=train[Y]
        - cross_val_predict(denoise_m, train[X], train[Y], cv=5)
        + train[Y].mean()
    )
    return train_pred_y


def plot_debiased_denoised(df):
    """Visualize debiased and denoised treatment"""
    np.random.seed(123)
    sns.scatterplot(data=df.sample(1000), x="price_res", y="sales_res", hue="weekday")
    return plt


# 4.) Use the orthogonalized residuals to estimate the causal effect (ATE), get negative relationship
def comparison_models(train_pred_y, train):
    """Compare the models"""
    final_model = smf.ols("sales_res~price_res", data=train_pred_y).fit().summary()
    # In coparison to unorthogonalized model, have positive relationship
    basic_model = smf.ols("sales~price", data=train).fit().summary()
    return final_model, basic_model


# ######################I.3 Parametric Double ML CATE###############################################
def parametric_double_ml_cate(train_pred_y, test):
    """Parametric in the second stage (still linear model), before fully interacted"""
    # #Now we want to produce the CATE, not the ATE, interact the residuals with the covariates
    final_model_cate = smf.ols(
        formula="sales_res ~ price_res * (temp + C(weekday) + cost)", data=train_pred_y
    ).fit()
    # Use that model to predict the CATE
    # Which is also called the R-Learner (Residualization)
    cate_test = test.assign(
        cate=final_model_cate.predict(test.assign(price_res=1))
        - final_model_cate.predict(test.assign(price_res=0))
    )
    return final_model_cate, cate_test


# #####################I.4 Non-parametric Double ML CATE############################################
# We still had linear specifications in the last step, now we want to use non-linear models
# Same as before:
def orthogonalize_treatment_and_outcome(train):
    """orthogonalize treatment and outcome, use cross validation for both models

    Args:
        train (dataFrame): training data

    Returns:
        train_pred_nonparam (dataFrame): training data with orthogonalized treatment and outcome
    """
    T = "price"
    X = ["temp", "weekday", "cost"]
    Y = "sales"
    debias_m = LGBMRegressor(max_depth=3)
    denoise_m = LGBMRegressor(max_depth=3)
    train_pred_nonparam = train.assign(
        price_res=train[T] - cross_val_predict(debias_m, train[X], train[T], cv=5),
        sales_res=train[Y] - cross_val_predict(denoise_m, train[X], train[Y], cv=5),
    )
    return train_pred_nonparam


def non_parametric_double_ml_cate(train_pred_nonparam, test):
    """Non-parametric in the second stage."""
    X = ["temp", "weekday", "cost"]
    # Define a causal loss function, that the R-learner minimizes
    model_final = LGBMRegressor(max_depth=3)
    # create the weights
    w = train_pred_nonparam["price_res"] ** 2
    # create the transformed target
    y_star = train_pred_nonparam["sales_res"] / train_pred_nonparam["price_res"]
    # use a weighted regression ML model to predict the target with the weights.
    model_final.fit(X=train_pred_nonparam[X], y=y_star, sample_weight=w)
    # Estimate the treatment effect
    cate_test_non_param = test.assign(cate=model_final.predict(test[X]))
    return cate_test_non_param
