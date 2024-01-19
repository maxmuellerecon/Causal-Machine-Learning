# Test file for G_Treatment_effect_estimators

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import pickle
from lightgbm import LGBMRegressor

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.G_Treatment_effect_estimators import (
    split_train_test,
    create_y_star,
    train_model,
    predict_cate,
    split_train_test_ice,
    predict_y_star_cont,
)


def test_split_train_test_shape_train():
    """Test split_train_test_ice."""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test(data, 0.4)
    assert train.shape == (9000, 8)


def test_split_train_test_shape_test():
    """Test split_train_test_ice."""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test(data, 0.4)
    assert test.shape == (6000, 8)


def test_create_y_star_shape():
    """Test create_y_star shape of y_star_train."""
    raw = pd.DataFrame(
        {
            "converted": [0, 1, 0, 1, 0],
            "em1": [0, 1, 1, 0, 1],
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "insurance": [0, 1, 0, 1, 0],
            "invested": [0, 1, 1, 0, 1],
        }
    )
    y_star_train, ps = create_y_star(raw)
    assert y_star_train.shape == (5,)


def test_create_y_star_type():
    """Test create_y_star type of y_star_train."""
    raw = pd.DataFrame(
        {
            "converted": [0, 1, 0, 1, 0],
            "em1": [0, 1, 1, 0, 1],
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "insurance": [0, 1, 0, 1, 0],
            "invested": [0, 1, 1, 0, 1],
        }
    )
    y_star_train, ps = create_y_star(raw)
    assert isinstance(y_star_train, pd.Series)


def test_create_y_star_ps_type():
    """Test create_y_star type of ps."""
    raw = pd.DataFrame(
        {
            "converted": [0, 1, 0, 1, 0],
            "em1": [0, 1, 1, 0, 1],
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "insurance": [0, 1, 0, 1, 0],
            "invested": [0, 1, 1, 0, 1],
        }
    )
    y_star_train, ps = create_y_star(raw)
    assert isinstance(ps, float)


def test_create_y_star_shape():
    """Test create_y_star shape of y_star_train."""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train, ps = create_y_star(train)
    assert y_star_train.shape == (9000,)


def test_create_y_star_ps_type():
    """Test create_y_star shape of ps."""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train, ps = create_y_star(train)
    assert isinstance(ps, float)


def test_train_model_type():
    """Test train_model type of cate_learner."""
    train_data = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "insurance": [0, 1, 0, 1, 0],
            "invested": [0, 1, 1, 0, 1],
        }
    )
    y_star_train = pd.Series([0, 1, 0, 1, 0])
    cate_learner = train_model(train_data, y_star_train)
    assert isinstance(cate_learner, LGBMRegressor)


def test_train_model_fit():
    """Test train_model fit."""
    train_data = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "insurance": [0, 1, 0, 1, 0],
            "invested": [0, 1, 1, 0, 1],
        }
    )
    y_star_train = pd.Series([0, 1, 0, 1, 0])
    cate_learner = train_model(train_data, y_star_train)
    assert hasattr(cate_learner, "fit")


def test_train_model_predict():
    """Test train_model predict."""
    train_data = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "insurance": [0, 1, 0, 1, 0],
            "invested": [0, 1, 1, 0, 1],
        }
    )
    y_star_train = pd.Series([0, 1, 0, 1, 0])
    cate_learner = train_model(train_data, y_star_train)
    test_data = pd.DataFrame(
        {
            "age": [50, 55, 60],
            "income": [100000, 110000, 120000],
            "insurance": [1, 0, 1],
            "invested": [1, 0, 0],
        }
    )
    predictions = cate_learner.predict(test_data)
    assert len(predictions) == 3


def test_train_model_type():
    """Test train_model type of cate_learner."""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train = pd.read_csv(
        BLD / "python" / "Lesson_G" / "data" / "y_star_train.csv"
    )
    cate_learner = train_model(train, y_star_train)
    assert isinstance(cate_learner, LGBMRegressor)


def test_predict_cate_shape():
    """Test predict_cate shape of test_pred."""
    test = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "test.csv")
    with open(BLD / "python" / "Lesson_G" / "model" / "cate_learner.pkl", "rb") as f:
        cate_learner = pickle.load(f)
    test_pred = predict_cate(test, cate_learner)
    actual = test_pred.shape
    expected = (6000, 9)
    assert actual == expected


def test_predict_cate_shape():
    """Test predict_cate shape of test_pred."""
    test = pd.DataFrame(
        {
            "age": [50, 55, 60],
            "income": [100000, 110000, 120000],
            "insurance": [1, 0, 1],
            "invested": [1, 0, 0],
        }
    )
    cate_learner = LGBMRegressor()
    cate_learner.fit(test, test["age"])
    test_pred = predict_cate(test, cate_learner)
    assert test_pred.shape == (3, 5)


def test_predict_cate_columns():
    """Test predict_cate columns of test_pred."""
    test = pd.DataFrame(
        {
            "age": [50, 55, 60],
            "income": [100000, 110000, 120000],
            "insurance": [1, 0, 1],
            "invested": [1, 0, 0],
        }
    )
    cate_learner = LGBMRegressor()
    cate_learner.fit(test, test["age"])
    test_pred = predict_cate(test, cate_learner)
    expected_columns = ["age", "income", "insurance", "invested", "cate"]
    assert list(test_pred.columns) == expected_columns


def test_predict_cate_type():
    """Test predict_cate type of test_pred."""
    test = pd.DataFrame(
        {
            "age": [50, 55, 60],
            "income": [100000, 110000, 120000],
            "insurance": [1, 0, 1],
            "invested": [1, 0, 0],
        }
    )
    cate_learner = LGBMRegressor()
    cate_learner.fit(test, test["age"])
    test_pred = predict_cate(test, cate_learner)
    assert isinstance(test_pred, pd.DataFrame)


def test_split_train_test_ice_shape_train():
    """Test split_train_test_ice shape of train."""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test_ice(data, 0.4)
    assert train.shape == (9000, 8)


def test_split_train_test_ice_shape_test():
    """Test split_train_test_ice shape of test."""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test_ice(data, 0.4)
    assert test.shape == (6000, 8)


def test_predict_y_star_cont_shape():
    """Test predict_y_star_cont shape of test_pred."""
    test = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "test_ice.csv")
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train_ice.csv")
    test_pred = predict_y_star_cont(train, test)
    actual = test_pred.shape
    expected = (1500, 6)
    assert actual == expected
