#Test file for G_Treatment_effect_estimators

import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import pickle
from lightgbm import LGBMRegressor

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.G_Treatment_effect_estimators import split_train_test, create_y_star, train_model, predict_cate, split_train_test_ice, predict_y_star_cont


def test_split_train_test_shape_train():
    """Test split_train_test_ice"""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test(data, 0.4)
    assert train.shape == (9000, 8)
    
def test_split_train_test_shape_test():
    """Test split_train_test_ice"""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test(data, 0.4)
    assert test.shape == (6000, 8)

def test_create_y_star_shape():
    """Test create_y_star shape of y_star_train"""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train, ps = create_y_star(train)
    assert y_star_train.shape == (9000,)
    
def test_create_y_star_type():
    """Test create_y_star shape of y_star_train"""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train, ps = create_y_star(train)
    assert isinstance(y_star_train, pd.Series)
    
def test_create_y_star_ps_type():
    """Test create_y_star shape of ps"""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train, ps = create_y_star(train)
    assert isinstance(ps, float)
    
def test_train_model_type():
    """Test train_model type of cate_learner"""
    train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train.csv")
    y_star_train = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "y_star_train.csv")
    cate_learner = train_model(train, y_star_train)
    assert isinstance(cate_learner, LGBMRegressor)
    
def test_predict_cate_shape():
    """Test predict_cate shape of test_pred"""
    test = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "test.csv")
    with open(BLD / "python" / "Lesson_G" / "model" / "cate_learner.pkl", "rb") as f:
        cate_learner = pickle.load(f)
    test_pred = predict_cate(test, cate_learner)
    actual = test_pred.shape
    expected = (6000, 9)
    assert actual == expected

def test_split_train_test_ice_shape_train():
    """Test split_train_test_ice shape of train"""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test_ice(data, 0.4)
    assert train.shape == (9000, 8)

def test_split_train_test_ice_shape_test():
    """Test split_train_test_ice shape of test"""
    data = pd.read_csv(SRC / "data" / "invest_email_rnd.csv")
    train, test = split_train_test_ice(data, 0.4)
    assert test.shape == (6000, 8)
    
def test_predict_y_star_cont_shape():
    """Test predict_y_star_cont shape of test_pred"""
    test = pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "test_ice.csv")
    train= pd.read_csv(BLD / "python" / "Lesson_G" / "data" / "train_ice.csv")
    test_pred = predict_y_star_cont(train, test)
    actual = test_pred.shape
    expected = (1500, 6)
    assert actual == expected