#Test file for A_Machine_learning_basics

import pandas as pd
import pickle
from matplotlib import pyplot as plt
import os

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.A_Machine_learning_basics import separate_customers, merge_customers, split_data, plot_profit_by_income, plot_profit_by_region, lower_bound_CI, plot_oos_performance, encode_region, specify_gradient_boosting_regressor, predict_net_value

def test_separate_customer_rows():
    """Test separate customer, amount of rows."""
    customer_transactions = pd.read_csv(SRC / "data" / "customer_transactions.csv")
    actual = separate_customers(customer_transactions)
    expected = 10000
    assert actual.shape[0] == expected

def test_separate_customer_columns():
    """Test separate customer, amount of columns."""
    customer_transactions = pd.read_csv(SRC / "data" / "customer_transactions.csv")
    actual = separate_customers(customer_transactions)
    expected = 2
    assert actual.shape[1] == expected

def test_separate_customer_unique():
    """Test separate customer, unique customers."""
    customer_transactions = pd.read_csv(SRC / "data" / "customer_transactions.csv")
    actual = separate_customers(customer_transactions)
    expected = 10000
    assert actual["customer_id"].nunique() == expected   

def test_merge_customers_rows():
    """Test merge customers, amount of rows."""
    profitable = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "profitable.dta")
    customer_features = pd.read_csv(SRC / "data" / "customer_features.csv")
    actual = merge_customers(profitable, customer_features)
    expected = 10000
    assert actual.shape[0] == expected
    
def test_split_data_rows():
    """Test split data, amount of rows."""
    customer_features_merged = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "customer_features_merged.dta")
    train_data, test_data = split_data(customer_features_merged, 0.3)
    actual = train_data.shape[0]
    expected = 7000
    assert actual == expected
    
def test_plot_profit_by_income_quantiles():
    """Check if the plot has bars corresponding to each income quantile"""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_income(train, 10)
    quantiles = 10
    actual_bars = len(fig.gca().patches)
    expected_bars = quantiles
    assert actual_bars == expected_bars
   
def test_plot_profit_by_income_title():
    """Check if the plot has the correct title"""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_income(train, 10)
    actual_title = fig.gca().get_title()
    expected_title = "Profitability by Income"
    assert actual_title == expected_title
    
def test_plot_profit_by_region_title():
    """Check if the plot has the correct title"""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_region(train)
    actual_title = fig.gca().get_title()
    expected_title = "Profitability by Region"
    assert actual_title == expected_title
    
def test_plot_profit_by_region_bars():
    """Check if the plot has bars corresponding to each region"""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_region(train)
    actual_bars = len(fig.gca().patches)
    expected_bars = 50
    assert actual_bars == expected_bars
    
def test_lower_bound_CI_return_types():
    """Check if the function returns a dictionary"""
    raw_data = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    result_net, result_invest = lower_bound_CI(raw_data)
    assert isinstance(result_net, dict)
    assert isinstance(result_invest, dict)
    
def test_plot_oos_performance_title():
    """Test if the function returns a plot"""
    rawtest = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "test.dta")
    regions = pickle.load(open(BLD / "python" / "Lesson_A" / "data" / "regions_to_invest.pkl", "rb"))
    region_policy = (rawtest[rawtest["region"].isin(regions.keys())])
    plot = plot_oos_performance(region_policy, regions)
    actual_title = plot.gca().get_title()
    expected_title = "Average Net Income: 41.56"
    assert actual_title == expected_title
    
def test_encode_region_return_types():
    """Test if the function returns a dictionary"""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    regions_to_net = pickle.load(open(BLD / "python" / "Lesson_A" / "data" / "regions_to_invest.pkl", "rb"))
    result = encode_region(train, regions_to_net)
    assert isinstance(result, pd.DataFrame)
            
def test_specify_gradient_boosting_regressor_path():
    """Test, if the function returns a pickle file"""
    train_encoded = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train_encoded.dta")
    model_features, reg_model = specify_gradient_boosting_regressor(train_encoded, 400, 4, 10, 0.01, 'squared_error')
    assert os.path.exists(BLD / "python" / "Lesson_A" / "models" / "reg_model.pkl")
    
def test_specify_gradient_boosting_regressor_return_types():
    """Test, if the function returns a dictionary"""
    train_encoded = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train_encoded.dta")
    model_features, reg_model = specify_gradient_boosting_regressor(train_encoded, 400, 4, 10, 0.01, 'squared_error')
    assert isinstance(model_features, pd.DataFrame)
    
def test_predict_net_value_train_r2_float():
    test = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "test.dta")
    reg_model = pickle.load(open(BLD / "python" / "Lesson_A" / "models" / "reg_model.pkl", "rb"))
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    regions_to_net = pickle.load(open(BLD / "python" / "Lesson_A" / "data" / "regions_to_net.pkl", "rb"))
    model_features = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "model_features.dta")
    model_policy, train_r2, test_r2 = predict_net_value(train, test, reg_model, regions_to_net, model_features)
    assert isinstance(train_r2, float)
    
def test_predict_net_value_test_r2_float():
    test = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "test.dta")
    reg_model = pickle.load(open(BLD / "python" / "Lesson_A" / "models" / "reg_model.pkl", "rb"))
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    regions_to_net = pickle.load(open(BLD / "python" / "Lesson_A" / "data" / "regions_to_net.pkl", "rb"))
    model_features = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "model_features.dta")
    model_policy, train_r2, test_r2 = predict_net_value(train, test, reg_model, regions_to_net, model_features)
    assert isinstance(test_r2, float)