#Exceptions and Fail functions

import pandas as pd
import numpy as np
from sklearn import ensemble
from keras import Sequential
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

###########################Fail Functions##############################################
def _fail_if_not_dataframe(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a dataframe, not %s" % type(data))
    
def _fail_if_too_many_quantiles(quantiles):
    if quantiles > 100:
        raise ValueError("Too many quantiles, please choose a number between 1 and 100")
    
def _fail_if_not_dict(regions):
    if not isinstance(regions, dict):
        raise TypeError("regions must be a dictionary, not %s" % type(regions))
                        
def _fail_if_irregular_learning_rate(learning_rate):
    if learning_rate > 1 or learning_rate < 0:
        raise ValueError("learning_rate must be between 0 and 1")
    
def _fail_if_not_gradient_boost(reg):
    if not isinstance(reg, ensemble._gb.GradientBoostingRegressor):
        raise TypeError("reg must be a Gradient Boosting Regressor, not %s" % type(reg))
    
def _fail_if_not_int(power):
    if not isinstance(power, int):
        raise TypeError("power must be an integer, not %s" % type(power))
    
def _fail_if_alpha_not_float_or_int(alpha):
    if not isinstance(alpha, float) and not isinstance(alpha, int):
        raise TypeError("alpha must be a float or integer, not %s" % type(alpha))
    
def _fail_if_not_Sequential(model):
    if not isinstance(model, Sequential):
        raise TypeError("model must be a Sequential, not %s" % type(model))
    
def _fail_if_not_list(data):
    if not isinstance(data, list):
        raise TypeError("data must be a list, not %s" % type(data))
    
def _fail_if_not_DecisionTreeRegressor(regressor):
    if not isinstance(regressor, DecisionTreeRegressor):
        raise TypeError("regressor must be a DecisionTreeRegressor, not %s" % type(regressor))
    
def _fail_if_not_array(data):
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array, not %s" % type(data))
    
def _fail_if_not_linear_regression(regressor):
    if not isinstance(regressor, smf.ols):
        raise TypeError("regressor must be a linear regression, not %s" % type(regressor))
    
def _fail_if_not_string(string):
    if not isinstance(string, str):
        raise TypeError("string must be a string, not %s" % type(string))
    
def _fail_if_not_between_zero_and_one(number):
    if number > 1 or number < 0:
        raise ValueError("number must be between 0 and 1")
    
def _fail_if_not_LGBMRegressor(regressor):
    if not isinstance(regressor, LGBMRegressor):
        raise TypeError("regressor must be a LGBMRegressor, not %s" % type(regressor))
    
def _fail_if_not_LogisticRegression(regressor):
    if not isinstance(regressor, LogisticRegression):
        raise TypeError("regressor must be a LogisticRegression, not %s" % type(regressor))