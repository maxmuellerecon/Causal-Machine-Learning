#Test file for F_Heterogeneous_treatment_effects

import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from matplotlib import pyplot as plt

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.F_Heterogeneous_treatment_effects import split_data, regressions_three, pred_sensitivity, ml_model, comparison, plot_regr_model, plot_ml_model

def test_split_data_shape_train():
    """Test split data shape."""
    data = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    train, test = split_data(data)
    actual = train.shape
    expected = (3500, 5)
    assert actual == expected

def test_split_data_shape_test():
    """Test split data shape."""
    data = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    train, test = split_data(data)
    actual = test.shape
    expected = (1500, 5)
    assert actual == expected
    
def test_regressions_three_number_params_m1():
    """Test regressions three shape for m1."""
    data = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    train, test = split_data(data)
    m1, m2, m3, latex_code1, latex_code2, latex_code3 = regressions_three(train)
    actual = m1.params.shape
    expected = (10,)
    assert actual == expected
    
def test_regressions_three_number_params_m2():
    """Test regressions three shape for m2."""
    data = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    train, test = split_data(data)
    m1, m2, m3, latex_code1, latex_code2, latex_code3 = regressions_three(train)
    actual = m2.params.shape
    expected = (11,)
    assert actual == expected
    
def test_regressions_three_number_params_m3():
    """Test regressions three shape for m3."""
    data = pd.read_csv(SRC / "data" / "ice_cream_sales_rnd.csv")
    train, test = split_data(data)
    m1, m2, m3, latex_code1, latex_code2, latex_code3 = regressions_three(train)
    actual = m3.params.shape
    expected = (18,)
    assert actual == expected
    
def test_pred_sensitivity_shape_m1():
    """Test pred sensitivity shape."""
    test = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "test.csv")
    with open(BLD / "python" / "Lesson_F" / "model" / "m1.pkl", "rb") as f:
        m1 = pickle.load(f)
    actual = pred_sensitivity(m1, test).shape
    expected = (1500, 7)
    assert actual == expected
    
def test_pred_sensitivity_shape_m3():
    """Test pred sensitivity shape."""
    test = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "test.csv")
    with open(BLD / "python" / "Lesson_F" / "model" / "m3.pkl", "rb") as f:
        m3 = pickle.load(f)
    actual = pred_sensitivity(m3, test).shape
    expected = (1500, 7)
    assert actual == expected
    
def test_ml_model_type_of_ml_model():
    """Test ml model type"""
    train = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "train.csv")
    test = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "test.csv")
    model = ml_model(train, test)
    assert isinstance(model, GradientBoostingRegressor)
    
def test_ml_model():
    """Test ml model n_estimators"""
    train = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "train.csv")
    test = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "test.csv")
    model = ml_model(train, test)
    actual = model.get_params()["n_estimators"]
    expected = 100
    assert actual == expected

def test_comparison_shape():
    """Test comparison shape."""
    regr_data = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "pred_sens_m3.csv")
    with open(BLD / "python" / "Lesson_F" / "model" / "m4.pkl", "rb") as f:
        m4 = pickle.load(f)
    actual = comparison(regr_data, m4, 2).shape
    expected = (1500, 11)
    assert actual == expected

def test_comparison_number_bands():
    regr_data = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "pred_sens_m3.csv")
    with open(BLD / "python" / "Lesson_F" / "model" / "m4.pkl", "rb") as f:
        m4 = pickle.load(f)
    bands_df = comparison(regr_data, m4, 2)
    actual = bands_df["sens_band"].nunique()
    expected = 2
    assert actual == expected
    
def test_plot_regr_model_plot_type():
    """Test plot regr model plot type."""
    regr_data = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "bands_df.csv")
    plot = plot_regr_model(regr_data)
    assert not isinstance(plot, plt.Axes)
    
def test_plot_ml_model_plot_type():
    """Test plot ml model plot type."""
    regr_data = pd.read_csv(BLD / "python" / "Lesson_F" / "data" / "bands_df.csv")
    plot = plot_ml_model(regr_data)
    assert not isinstance(plot, plt.Axes)