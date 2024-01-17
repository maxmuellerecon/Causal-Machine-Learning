#Test file for H_Meta_learners

import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import pickle
from sklearn.linear_model import LogisticRegression

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.H_Meta_learners import s_learner_fit, s_learner_predict, s_learner_evaluate_model, t_learner_fit, t_learner_cate, x_learner_fit, propensity_score_model, x_learner_st2, ps_predict, apply_ps_predict

def test_s_learner_fit_type():
    """Test s_learner_fit type of s_learner"""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    s_learner = s_learner_fit(data, 3, 30)
    assert isinstance(s_learner, LGBMRegressor)
    
def test_s_learner_predict_type_s_learner_cate_train():
    """Test s_learner_predict type of s_learner_cate_train"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    with open(BLD / "python" / "Lesson_H" / "model" / "s_learner.pkl", "rb") as f:
        s_learner = pickle.load(f)
    s_learner_cate_train, s_learner_cate_test = s_learner_predict(train, test, s_learner)
    assert isinstance(s_learner_cate_train, np.ndarray)
    
def test_s_learner_predict_type_s_learner_cate_test():
    """Test s_learner_predict type of s_learner_cate_train"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    with open(BLD / "python" / "Lesson_H" / "model" / "s_learner.pkl", "rb") as f:
        s_learner = pickle.load(f)
    s_learner_cate_train, s_learner_cate_test = s_learner_predict(train, test, s_learner)
    assert isinstance(s_learner_cate_test, pd.DataFrame)
    
def test_s_learner_evaluate_model_mse_type():
    """Test s_learner_evaluate_model"""
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    s_learner_cate_test = pd.read_csv(BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_test.csv")
    mse, mae = s_learner_evaluate_model(test, s_learner_cate_test)
    assert isinstance(mse, float)
    
def test_s_learner_evaluate_model_mae_type():
    """Test s_learner_evaluate_model"""
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    s_learner_cate_test = pd.read_csv(BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_test.csv")
    mse, mae = s_learner_evaluate_model(test, s_learner_cate_test)  
    assert isinstance(mae, float)
    
def test_t_learner_fit_type_t_m0():
    """Test t_learner_fit type of t_m0"""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    t_m0, t_m1 = t_learner_fit(data, 3, 30)
    assert isinstance(t_m0, LGBMRegressor)

def test_t_learner_fit_type_t_m1():
    """Test t_learner_fit type of t_m1"""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    t_m0, t_m1 = t_learner_fit(data, 3, 30)
    assert isinstance(t_m1, LGBMRegressor)
    
def test_t_learner_cate_type_t_learner_cate_train():
    """Test t_learner_cate type of t_learner_cate_train"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    t_m0, t_m1 = t_learner_fit(train, 3, 30)
    t_learner_cate_train, t_learner_cate_test = t_learner_cate(train, test, t_m0, t_m1)
    assert isinstance(t_learner_cate_train, np.ndarray)
    
def test_t_learner_cate_type_t_learner_cate_test():
    """Test t_learner_cate type of t_learner_cate_test"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    t_m0, t_m1 = t_learner_fit(train, 3, 30)
    t_learner_cate_train, t_learner_cate_test = t_learner_cate(train, test, t_m0, t_m1)
    assert isinstance(t_learner_cate_test, pd.DataFrame)
    
def test_x_learner_fit_type_x1_m0():
    """Test x_learner_fit type of x1_m0"""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(data, 3, 30)
    assert isinstance(x1_m0, LGBMRegressor)
    
def test_x_learner_fit_type_x1_m1():
    """Test x_learner_fit type of x1_m1"""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1= x_learner_fit(data, 3, 30)
    assert isinstance(x1_m1, LGBMRegressor)
    
def test_propensity_score_model_type_g():
    """Test propensity_score_model type of g"""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    g = propensity_score_model(data)
    assert isinstance(g, LogisticRegression)

def test_x_learner_st2_type_x2_m0():
    """Test x_learner_st2 type of x2_m0"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    assert isinstance(x2_m0, LGBMRegressor)
    
def test_x_learner_st2_type_x2_m1():
    """Test x_learner_st2 type of x2_m1"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    assert isinstance(x2_m1, LGBMRegressor)
    
def test_x_learner_st2_type_d_train():
    """Test x_learner_st2 type of d_train"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    assert isinstance(d_train, np.ndarray)
    
def test_ps_predict_type():
    """Test ps_predict type of ps_predict"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    g = propensity_score_model(train)
    ps_predict(g, train, 1)
    assert isinstance(ps_predict(g, train, 1), np.ndarray)
    
def test_apply_ps_predict_type_x_learner_cate_train():
    """Test apply_ps_predict type of x_learner_cate_train"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    g = propensity_score_model(train)
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    x_learner_cate_train, x_learner_cate_test = apply_ps_predict(train, test, g, x2_m0, x2_m1)
    assert isinstance(x_learner_cate_train, np.ndarray)
    
def test_apply_ps_predict_type_x_learner_cate_test():
    """Test apply_ps_predict type of x_learner_cate_test"""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    g = propensity_score_model(train)
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    x_learner_cate_train, x_learner_cate_test = apply_ps_predict(train, test, g, x2_m0, x2_m1)
    assert isinstance(x_learner_cate_test, pd.DataFrame)