#Testing file for J_Diff_in_diff_and_ml

import pytest
pytestmark = pytest.mark.filterwarnings("ignore")
import pandas as pd
import statsmodels
import pickle

from causalmachinelearning.config import BLD
from causalmachinelearning.lessons.J_Diff_and_diff_and_ml import create_data, plot_trend, twfe_regression, plot_counterfactuals, fct_late_vs_early, plot_twfe_regression_late_vs_early, twfe_regression_groups, check_trueATT_vs_predATT, plot_treatment_effect

def test_create_data_type():
    """Test type of output"""
    data = create_data()
    assert isinstance(data, pd.DataFrame)
    
def test_create_data_shape():
    """Test shape of output"""
    data = create_data()
    assert data.shape == (9200, 14)
    
def test_plot_trend_type():
    """Test type of output"""
    data = create_data()
    plt = plot_trend(data)
    assert isinstance(plt, type(plt))
    
def test_twfe_regression_type():
    """Test type of output"""
    data = create_data()
    twfe_output, twfe_model = twfe_regression(data)
    assert isinstance(twfe_model, statsmodels.regression.linear_model.RegressionResultsWrapper)
    
def test_plot_counterfactuals_type():
    """Test type of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    twfe_output, twfe_model = twfe_regression(data)
    plt = plot_counterfactuals(data, twfe_model)
    assert isinstance(plt, type(plt))
    
def test_cohort_unique():
    """Test if the cohort variable is unique"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    expected = 3
    unique_cohorts = data["cohort"].unique()
    assert len(unique_cohorts) == expected
    
def test_fct_late_vs_early_type():
    """Test type of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    data_late_vs_early_data = fct_late_vs_early(data)
    assert isinstance(data_late_vs_early_data, pd.DataFrame)

def test_fct_late_vs_early_shape():
    """Test type of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    data_late_vs_early_data = fct_late_vs_early(data)
    assert data_late_vs_early_data.shape == (4392, 14)
    
def test_plot_twfe_regression_late_vs_early_type():
    """Test type of output"""
    data_late_vs_early = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data_late_vs_early.csv")
    plt = plot_twfe_regression_late_vs_early(data_late_vs_early)
    assert isinstance(plt, type(plt))
    
def test_twfe_regression_groups_type_data():
    """Test type of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    twfe_model_groups, df_heter_str = twfe_regression_groups(data)
    assert isinstance (df_heter_str, pd.DataFrame)
    
def test_twfe_regression_groups_shape_data():
    """Test shape of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    twfe_model_groups, df_heter_str = twfe_regression_groups(data)
    assert df_heter_str.shape == (9200, 14)

def test_twfe_regression_groups_type_model():
    """Test type of model"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    twfe_model_groups, df_heter_str = twfe_regression_groups(data)
    assert isinstance (twfe_model_groups, statsmodels.regression.linear_model.RegressionResultsWrapper)

def test_check_trueATT_vs_predATT_type_data():
    """Test type of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    with open(BLD / "python" / "Lesson_J" / "model" / "TWFE_model_groups.pkl", "rb") as f:
        twfe_model_groups = pickle.load(f)
    df_pred, length, tau_mean, pred_effect_mean = check_trueATT_vs_predATT(data, twfe_model_groups)
    assert isinstance (df_pred, pd.DataFrame)
    
def test_check_trueATT_vs_predATT_shape_data():
    """Test shape of output"""
    data = pd.read_csv(BLD / "python" / "Lesson_J" / "data" / "data.csv")
    with open(BLD / "python" / "Lesson_J" / "model" / "TWFE_model_groups.pkl", "rb") as f:
        twfe_model_groups = pickle.load(f)
    df_pred, length, tau_mean, pred_effect_mean = check_trueATT_vs_predATT(data, twfe_model_groups)
    assert df_pred.shape == (9200, 16)
    
def test_plot_treatment_effect_type():
    """Test type of output"""
    with open(BLD / "python" / "Lesson_J" / "model" / "TWFE_model_groups.pkl", "rb") as f:
        twfe_model_groups = pickle.load(f)
    plt = plot_treatment_effect(twfe_model_groups)
    assert isinstance(plt, type(plt))