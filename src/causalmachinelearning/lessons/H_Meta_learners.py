# H_Meta_learners

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression

from causalmachinelearning.lessons.__exceptions import _fail_if_not_dataframe, _fail_if_not_int, _fail_if_not_LGBMRegressor, _fail_if_not_list, _fail_if_not_array, _fail_if_not_LogisticRegression

# Meta Learners: Use predictive machine learning (here Gradient Boosted Trees) to estimate the CATE
# non-random data to train the meta learner --> harder because we need to de-bias the data and we need to estimate the CATE
# random data to validate the meta learner


#####################H.1 S-learner#####################
# Simple learner, single model
# Positive: Can deal with continuous and discrete treatments
# Negative: Can bias treatment effects towards zero, depending on level of regularization in model and it might discard treatment variable if it is weak
def s_learner_fit(raw, depth, samples):
    """give out the s_learner, but include the treatment variable in the X"""
    _fail_if_not_dataframe(raw)
    _fail_if_not_int(depth)
    _fail_if_not_int(samples)
    
    Y = "converted"
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    np.random.seed(123)
    s_learner = LGBMRegressor(max_depth=depth, min_child_samples=samples)
    s_learner.fit(raw[X + [T]], raw[Y])
    return s_learner


def s_learner_predict(train, test, s_learner):
    """Use prediction of s-learner to predict the cate for the test data and then assign it to the test data, labelling it cate"""
    _fail_if_not_dataframe(train)
    _fail_if_not_dataframe(test)
    _fail_if_not_LGBMRegressor(s_learner)
    
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    # Calculates the treatment effect by taking the difference in the predicted outcomes between the treated and untreated groups.
    s_learner_cate_train = s_learner.predict(
        train[X].assign(**{T: 1})
    ) - s_learner.predict(train[X].assign(**{T: 0}))
    # Use prediction of s-learner to predict the cate for the test data and then assign it to the test data, labelling it cate
    s_learner_cate_test = test.assign(
        cate=(
            s_learner.predict(test[X].assign(**{T: 1}))
            - s_learner.predict(test[X].assign(**{T: 0}))  # predict under treatment
        )  # predict under control
    )
    return s_learner_cate_train, s_learner_cate_test


def s_learner_evaluate_model(test, s_learner_cate_test):
    """Evaluate the model"""
    _fail_if_not_dataframe(test)
    _fail_if_not_dataframe(s_learner_cate_test)
    
    true_cate_test = test["em1"]
    mse = mean_squared_error(true_cate_test, s_learner_cate_test["cate"])
    mae = mean_absolute_error(true_cate_test, s_learner_cate_test["cate"])
    return mse, mae


# #####################H.2 T-learner#####################
# #Use one model per treatment variable
# #Positive: Does not discount treatment variable, we split on it
# #Negative: Can only deal with binary treatments and still suffers from regularization bias - penalty that introduces bias, but gains in variance; needs larger sample
def t_learner_fit(train_data, depth, samples):
    """give out the two models, split by treatment variable"""
    _fail_if_not_dataframe(train_data)
    _fail_if_not_int(depth)
    _fail_if_not_int(samples)
    
    np.random.seed(123)
    t_m0 = LGBMRegressor(max_depth=depth, min_child_samples=samples)
    t_m1 = LGBMRegressor(max_depth=depth, min_child_samples=samples)
    Y = "converted"
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    t_m0.fit(train_data.query(f"{T}==0")[X], train_data.query(f"{T}==0")[Y])
    t_m1.fit(train_data.query(f"{T}==1")[X], train_data.query(f"{T}==1")[Y])
    return t_m0, t_m1


def t_learner_cate(train, test, t_m0, t_m1):
    """Use prediction of t-learner to predict the cate for the test data and then assign it to the test data, labelling it cate"""
    _fail_if_not_dataframe(train)
    _fail_if_not_dataframe(test)
    _fail_if_not_LGBMRegressor(t_m0)
    _fail_if_not_LGBMRegressor(t_m1)
    
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    # Calculates the treatment effect by taking the difference in the predicted outcomes between the treated and untreated groups.
    t_learner_cate_train = t_m1.predict(train[X]) - t_m0.predict(train[X])
    t_learner_cate_test = test.assign(
        cate=(t_m1.predict(test[X]) - t_m0.predict(test[X]))
    )
    return t_learner_cate_train, t_learner_cate_test


# #####################H.3 X-learner#####################
# #1.Stage: Split the sample again in two parts
# #2.Stage: Impute the treatment effect for the control and for the treated and fit two more models to predict the treatment effects in the two models.
# #3.Stage: Use propensity scores to weight the two models and combine them to get the final treatment effect - give more weight to the CATE model that was estimated where the assigned treatment was more likely

# #Positive: Does a good job with non-linear treatment effects, performs better when sizes of the two groups are different, hence small treatment group
# #Negative: only discrete treatments


# 1. first stage models - like t lerner
def x_learner_fit(train_data, depth, samples):
    """Give out the two models, split by treatment variable"""
    _fail_if_not_dataframe(train_data)
    _fail_if_not_int(depth)
    _fail_if_not_int(samples)
    
    np.random.seed(123)
    Y = "converted"
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    x1_m0 = LGBMRegressor(max_depth=depth, min_child_samples=samples)
    x1_m1 = LGBMRegressor(max_depth=depth, min_child_samples=samples)
    # Fit models on the training data, split by treatment variable
    x1_m0.fit(train_data.query(f"{T}==0")[X], train_data.query(f"{T}==0")[Y])
    x1_m1.fit(train_data.query(f"{T}==1")[X], train_data.query(f"{T}==1")[Y])
    return x1_m0, x1_m1


def propensity_score_model(train_data):
    """Give out the propensity score model"""
    _fail_if_not_dataframe(train_data)
    
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    g = LogisticRegression(solver="lbfgs", penalty="none")
    g.fit(train_data[X], train_data[T])
    return g


# 2. impute the treatment effect and fit the second stage models on them.
# prediction - true Y in both models
def x_learner_st2(train_data, x1_m0, x1_m1):
    """Train second stage of the x_learner model and save it as a pickle file"""
    _fail_if_not_dataframe(train_data)
    _fail_if_not_LGBMRegressor(x1_m0)
    _fail_if_not_LGBMRegressor(x1_m1)
    
    Y = "converted"
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    d_train = np.where(
        train_data[T] == 0,
        x1_m1.predict(train_data[X]) - train_data[Y],
        train_data[Y] - x1_m0.predict(train_data[X]),
    )
    # Second stage
    x2_m0 = LGBMRegressor(max_depth=2, min_child_samples=30)
    x2_m1 = LGBMRegressor(max_depth=2, min_child_samples=30)
    # Train the second stage models on the imputed treatment effects, split by treatment variable
    x2_m0.fit(train_data.query(f"{T}==0")[X], d_train[train_data[T] == 0])
    x2_m1.fit(train_data.query(f"{T}==1")[X], d_train[train_data[T] == 1])
    return x2_m0, x2_m1, d_train


# 3.) Use propensity score model
def ps_predict(g, df, t):
    """Predict the propensity score based on covariates X"""
    _fail_if_not_dataframe(df)
    _fail_if_not_int(t)
    _fail_if_not_LogisticRegression(g)
    
    X = ["age", "income", "insurance", "invested"]
    return g.predict_proba(df[X])[:, t]


def apply_ps_predict(train, test, g, x2_m0, x2_m1):
    """Get the CATE of the x learner for the train and test dataset

    Args:
        train (DataFrame): train data
        test (DataFrame): test data
        g (LogisticRegressor): Logistic regression model for ps
        x2_m0 (LGBMRegressor): m0 model
        x2_m1 (LGBMRegressor): m1 model

    Returns:
        x_learner_cate_train(DataFrame): Data with CATES for train data
        x_learner_cate_test (DataFrame): Data with CATES for test data
    """
    _fail_if_not_dataframe(train)
    _fail_if_not_dataframe(test)
    _fail_if_not_LogisticRegression(g)
    _fail_if_not_LGBMRegressor(x2_m0)
    _fail_if_not_LGBMRegressor(x2_m1)
    
    X = ["age", "income", "insurance", "invested"]
    # Weight the 2 models by the propensity score
    x_learner_cate_train = ps_predict(g, train, 1) * x2_m0.predict(
        train[X]
    ) + ps_predict(g, train, 0) * x2_m1.predict(train[X])
    # Use it on the test data to predict the cate
    x_learner_cate_test = test.assign(
        cate=(
            ps_predict(g, test, 1) * x2_m0.predict(test[X])
            + ps_predict(g, test, 0) * x2_m1.predict(test[X])
        )
    )
    return x_learner_cate_train, x_learner_cate_test
