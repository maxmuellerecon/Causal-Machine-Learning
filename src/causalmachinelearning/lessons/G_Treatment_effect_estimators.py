#G_Treatment_effect_estimators

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

################################G.1 From Outcomes to treatment effects################################
#From predicting outcomes to predicting treatment effects, called F-Learner
#Works only for discrete or binary treatments

#Set up train and test
def split_train_test(raw, size):
    """Split sample into train and test"""
    np.random.seed(123)
    train, test = train_test_split(raw, test_size=size)
    return train, test


#Set up y_star
def create_y_star(raw):
    """Create y_star, the transformed outcome variable

    Args:
        raw (DataFrame): The training data

    Returns:
        y_star_train (DataFrame): y_star_train, the transformed outcome
        ps (float): propensity score
    """
    y = "converted"
    T = "em1"
    X = ["age", "income", "insurance", "invested"]
    ps = raw[T].mean()
    y_star_train = raw[y] * (raw[T] - ps)/(ps*(1-ps))
    return y_star_train, ps


#Use boosted trees to predict the transformed outcome y_star
def train_model(tr_data, y_star_train):
    """Train the model via boosted trees

    Args:
        tr_data (DataFrame): The training data
        y_star_train (DataFrame): y_star_train, the transformed outcome

    Returns:
        cate_learner (pkl): ML model that is trained
    """
    np.random.seed(123)
    X = ["age", "income", "insurance", "invested"]
    cate_learner = LGBMRegressor(max_depth=3, min_child_samples=300, num_leaves=5)
    cate_learner.fit(tr_data[X], y_star_train)
    return cate_learner


def predict_cate(test, cate_learner):
    """Predict cate for test data based on the trained model via boosted trees

    Args:
        test (DataFrame): The test data
        cate_learner (LGBMRegressor): The trained model

    Returns:
        test_pred (DataFrame): The test data with predicted cate
    """
    #Predict cate for test data based on the trained model via boosted trees
    X = ["age", "income", "insurance", "invested"]
    test_pred = test.assign(cate=cate_learner.predict(test[X]))
    return test_pred


# ################################G.2 Continuous Treatment################################
#Set up train and test
def split_train_test_ice(raw, size):
    """Split sample into train and test"""
    np.random.seed(123)
    train, test = train_test_split(raw, test_size=size)
    return train, test


#Fit a model to predict y_star (treatment effect)
def predict_y_star_cont(tr_data, test_data):
    """Predict y star for continuous treatment

    Args:
        tr_data (DataFrame): training data
        test_data (DataFrame): test data

    Returns:
        test_pred (DataFrame): test data with predicted y_star (CATE)
    """
    y_star_cont = ((tr_data["price"] - tr_data["price"].mean())
               *(tr_data["sales"] - tr_data["sales"].mean()))
    cate_learner = LGBMRegressor(max_depth=3, min_child_samples=300, num_leaves=5)
    np.random.seed(123)
    cate_learner.fit(tr_data[["temp", "weekday", "cost"]], y_star_cont)                 #Fit the model on train data
    cate_test_transf_y = cate_learner.predict(test_data[["temp", "weekday", "cost"]])        #predict cate for test data 
    test_pred = test_data.assign(cate=cate_test_transf_y)                                    #assign test cate to test data
    test_pred.sample(5)
    return test_pred