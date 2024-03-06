# Test file for H_Meta_learners

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.H_Meta_learners import (
    s_learner_fit,
    s_learner_predict,
    s_learner_evaluate_model,
    t_learner_fit,
    t_learner_cate,
    x_learner_fit,
    propensity_score_model,
    x_learner_st2,
    ps_predict,
    apply_ps_predict,
)


def test_s_learner_fit_type():
    """Test s_learner_fit type of s_learner."""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    s_learner = s_learner_fit(data, 3, 30)
    assert isinstance(s_learner, LGBMRegressor)


def test_s_learner_predict_type_s_learner_cate_train():
    """Test s_learner_predict type of s_learner_cate_train."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    with open(BLD / "python" / "Lesson_H" / "model" / "s_learner.pkl", "rb") as f:
        s_learner = pickle.load(f)
    s_learner_cate_train, s_learner_cate_test = s_learner_predict(
        train, test, s_learner
    )
    assert isinstance(s_learner_cate_train, np.ndarray)


def test_s_learner_predict_type_s_learner_cate_test():
    """Test s_learner_predict type of s_learner_cate_train."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    with open(BLD / "python" / "Lesson_H" / "model" / "s_learner.pkl", "rb") as f:
        s_learner = pickle.load(f)
    s_learner_cate_train, s_learner_cate_test = s_learner_predict(
        train, test, s_learner
    )
    assert isinstance(s_learner_cate_test, pd.DataFrame)


def test_s_learner_evaluate_model_mse():
    """Test s_learner_evaluate_model."""
    test_data = pd.DataFrame(
        {"em1": [1, 2, 3, 4, 5], "cate": [1.1, 2.2, 3.3, 4.4, 5.5]}
    )
    true_cate_test = test_data["em1"]
    s_learner_cate_test = test_data[["cate"]]
    mse, mae = s_learner_evaluate_model(test_data, s_learner_cate_test)
    assert np.isclose(
        mse, mean_squared_error(true_cate_test, s_learner_cate_test["cate"])
    )


def test_s_learner_evaluate_model_mae():
    """Test s_learner_evaluate_model."""
    test_data = pd.DataFrame(
        {"em1": [1, 2, 3, 4, 5], "cate": [1.1, 2.2, 3.3, 4.4, 5.5]}
    )
    true_cate_test = test_data["em1"]
    s_learner_cate_test = test_data[["cate"]]
    mse, mae = s_learner_evaluate_model(test_data, s_learner_cate_test)
    assert np.isclose(
        mae, mean_absolute_error(true_cate_test, s_learner_cate_test["cate"])
    )


def test_s_learner_evaluate_model_mse_type():
    """Test s_learner_evaluate_model."""
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    s_learner_cate_test = pd.read_csv(
        BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_test.csv"
    )
    mse, mae = s_learner_evaluate_model(test, s_learner_cate_test)
    assert isinstance(mse, float)


def test_s_learner_evaluate_model_mae_type():
    """Test s_learner_evaluate_model."""
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    s_learner_cate_test = pd.read_csv(
        BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_test.csv"
    )
    mse, mae = s_learner_evaluate_model(test, s_learner_cate_test)
    assert isinstance(mae, float)


def test_t_learner_fit_type_t_m0():
    """Test t_learner_fit type of t_m0."""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    t_m0, t_m1 = t_learner_fit(data, 3, 30)
    assert isinstance(t_m0, LGBMRegressor)


def test_t_learner_fit_type_t_m1():
    """Test t_learner_fit type of t_m1."""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    t_m0, t_m1 = t_learner_fit(data, 3, 30)
    assert isinstance(t_m1, LGBMRegressor)


def test_t_learner_cate_type_t_learner_cate_train():
    """Test t_learner_cate type of t_learner_cate_train."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    t_m0, t_m1 = t_learner_fit(train, 3, 30)
    t_learner_cate_train, t_learner_cate_test = t_learner_cate(train, test, t_m0, t_m1)
    assert isinstance(t_learner_cate_train, np.ndarray)


def test_t_learner_cate_type_t_learner_cate_test():
    """Test t_learner_cate type of t_learner_cate_test."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    t_m0, t_m1 = t_learner_fit(train, 3, 30)
    t_learner_cate_train, t_learner_cate_test = t_learner_cate(train, test, t_m0, t_m1)
    assert isinstance(t_learner_cate_test, pd.DataFrame)


def test_x_learner_fit_type_x1_m0():
    """Test x_learner_fit type of x1_m0."""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(data, 3, 30)
    assert isinstance(x1_m0, LGBMRegressor)


def test_x_learner_fit_type_x1_m1():
    """Test x_learner_fit type of x1_m1."""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(data, 3, 30)
    assert isinstance(x1_m1, LGBMRegressor)


def test_propensity_score_model_type_g():
    """Test propensity_score_model type of g."""
    data = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    g = propensity_score_model(data)
    assert isinstance(g, LogisticRegression)


def test_x_learner_st2_type_x2_m0():
    """Test x_learner_st2 type of x2_m0."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    assert isinstance(x2_m0, LGBMRegressor)


def test_x_learner_st2_type_x2_m1():
    """Test x_learner_st2 type of x2_m1."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    assert isinstance(x2_m1, LGBMRegressor)


def test_x_learner_st2_type_d_train():
    """Test x_learner_st2 type of d_train."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    assert isinstance(d_train, np.ndarray)


def test_ps_predict_treatment_0():
    """Test ps_predict."""
    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "insurance": [1, 0, 1],
            "invested": [0, 1, 1],
        }
    )
    # Create a sample LogisticRegression model
    g = LogisticRegression()
    g.fit(df[["age", "income", "insurance", "invested"]], df["invested"])
    # Test propensity score prediction for treatment t=0
    t = 0
    expected_scores_t0 = np.array([0.90519076, 0.13599325, 0.00133255])
    scores_t0 = ps_predict(g, df, t)
    assert np.allclose(scores_t0, expected_scores_t0)


def test_ps_predict_treatment_1():
    """Test ps_predict."""
    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "insurance": [1, 0, 1],
            "invested": [0, 1, 1],
        }
    )
    # Create a sample LogisticRegression model
    g = LogisticRegression()
    g.fit(df[["age", "income", "insurance", "invested"]], df["invested"])
    # Test propensity score prediction for treatment t=1
    t = 1
    expected_scores_t1 = np.array([0.09480924, 0.86400675, 0.99866745])
    scores_t1 = ps_predict(g, df, t)
    assert np.allclose(scores_t1, expected_scores_t1)


def test_ps_predict_type():
    """Test ps_predict type of ps_predict."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    g = propensity_score_model(train)
    ps_predict(g, train, 1)
    assert isinstance(ps_predict(g, train, 1), np.ndarray)


def test_apply_ps_predict_type_x_learner_cate_train():
    """Test apply_ps_predict type of x_learner_cate_train."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    g = propensity_score_model(train)
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    x_learner_cate_train, x_learner_cate_test = apply_ps_predict(
        train, test, g, x2_m0, x2_m1
    )
    assert isinstance(x_learner_cate_train, np.ndarray)


def test_apply_ps_predict_type_x_learner_cate_test():
    """Test apply_ps_predict type of x_learner_cate_test."""
    train = pd.read_csv(SRC / "data" / "invest_email_biased.csv")
    test = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    g = propensity_score_model(train)
    x1_m0, x1_m1 = x_learner_fit(train, 3, 30)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    x_learner_cate_train, x_learner_cate_test = apply_ps_predict(
        train, test, g, x2_m0, x2_m1
    )
    assert isinstance(x_learner_cate_test, pd.DataFrame)
