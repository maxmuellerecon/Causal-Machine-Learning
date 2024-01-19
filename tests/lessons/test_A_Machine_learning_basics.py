# Test file for A_Machine_learning_basics

import pandas as pd
import pickle
import os

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.A_Machine_learning_basics import (
    separate_customers,
    merge_customers,
    split_data,
    plot_profit_by_income,
    plot_profit_by_region,
    lower_bound_CI,
    plot_oos_performance,
    encode_region,
    specify_gradient_boosting_regressor,
    predict_net_value,
)


def test_separate_customers_expected_output():
    """Test separate_customers function."""
    # Create a sample raw dataset
    raw = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "transaction_1": [10, 20, 30],
            "transaction_2": [5, 15, 25],
        }
    )
    # Call the separate_customers function
    actual = separate_customers(raw)
    # Define the expected output
    expected = pd.DataFrame({"customer_id": [1, 2, 3], "net_value": [15, 35, 55]})
    # Compare the actual and expected outputs
    pd.testing.assert_frame_equal(actual, expected)


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


def test_merge_customers_expected_output():
    """Test merge_customers function."""
    # Create a sample raw dataset
    raw = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "transaction_1": [10, 20, 30],
            "transaction_2": [5, 15, 25],
        }
    )
    # Create a sample merge_data dataset
    merge_data = pd.DataFrame(
        {"customer_id": [1, 2, 3], "age": [25, 30, 35], "gender": ["M", "F", "M"]}
    )
    # Call the merge_customers function
    actual = merge_customers(raw, merge_data)
    # Define the expected output
    expected = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "age": [25, 30, 35],
            "gender": ["M", "F", "M"],
            "transaction_1": [10, 20, 30],
            "transaction_2": [5, 15, 25],
        }
    )
    # Compare the actual and expected outputs
    pd.testing.assert_frame_equal(actual, expected)


def test_merge_customers_rows():
    """Test merge customers, amount of rows."""
    profitable = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "profitable.dta")
    customer_features = pd.read_csv(SRC / "data" / "customer_features.csv")
    actual = merge_customers(profitable, customer_features)
    expected = 10000
    assert actual.shape[0] == expected


def test_split_data_rows():
    """Test split data, amount of rows."""
    customer_features_merged = pd.read_stata(
        BLD / "python" / "Lesson_A" / "data" / "customer_features_merged.dta"
    )
    train_data, test_data = split_data(customer_features_merged, 0.3)
    actual = train_data.shape[0]
    expected = 7000
    assert actual == expected


def test_plot_profit_by_income_quantiles():
    """Check if the plot has bars corresponding to each income quantile."""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_income(train, 10)
    quantiles = 10
    actual_bars = len(fig.gca().patches)
    expected_bars = quantiles
    assert actual_bars == expected_bars


def test_plot_profit_by_income_title():
    """Check if the plot has the correct title."""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_income(train, 10)
    actual_title = fig.gca().get_title()
    expected_title = "Profitability by Income"
    assert actual_title == expected_title


def test_plot_profit_by_region_title():
    """Check if the plot has the correct title."""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_region(train)
    actual_title = fig.gca().get_title()
    expected_title = "Profitability by Region"
    assert actual_title == expected_title


def test_plot_profit_by_region_bars():
    """Check if the plot has bars corresponding to each region."""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    fig = plot_profit_by_region(train)
    actual_bars = len(fig.gca().patches)
    expected_bars = 50
    assert actual_bars == expected_bars


def test_lower_bound_CI_expected_output_regions_to_net():
    """Test lower_bound_CI function."""
    # Create a sample raw dataset
    raw = pd.DataFrame(
        {
            "region": ["A", "A", "B", "B", "C", "C"],
            "net_value": [10, 20, 30, 40, 50, 60],
        }
    )
    # Call the lower_bound_CI function
    regions_to_net, regions_to_invest = lower_bound_CI(raw)
    # Define the expected output
    expected_regions_to_net = {"A": 15.0, "B": 35.0, "C": 55.0}
    expected_regions_to_invest = {"B": 35.0, "C": 55.0}
    # Compare the actual and expected outputs
    assert regions_to_net == expected_regions_to_net


def test_lower_bound_CI_return_types():
    """Check if the function returns a dictionary."""
    raw_data = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    result_net, result_invest = lower_bound_CI(raw_data)
    assert isinstance(result_net, dict)
    assert isinstance(result_invest, dict)


def test_plot_oos_performance_title():
    """Test if the function returns a plot."""
    rawtest = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "test.dta")
    regions = pickle.load(
        open(BLD / "python" / "Lesson_A" / "data" / "regions_to_invest.pkl", "rb")
    )
    region_policy = rawtest[rawtest["region"].isin(regions.keys())]
    plot = plot_oos_performance(region_policy, regions)
    actual_title = plot.gca().get_title()
    expected_title = "Average Net Income: 41.56"
    assert actual_title == expected_title


def test_encode_region_return_types():
    """Test if the function returns a dictionary."""
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    regions_to_net = pickle.load(
        open(BLD / "python" / "Lesson_A" / "data" / "regions_to_invest.pkl", "rb")
    )
    result = encode_region(train, regions_to_net)
    assert isinstance(result, pd.DataFrame)


def test_specify_gradient_boosting_regressor_path():
    """Test, if the function returns a pickle file."""
    train_encoded = pd.read_stata(
        BLD / "python" / "Lesson_A" / "data" / "train_encoded.dta"
    )
    model_features, reg_model = specify_gradient_boosting_regressor(
        train_encoded, 400, 4, 10, 0.01, "squared_error"
    )
    assert os.path.exists(BLD / "python" / "Lesson_A" / "models" / "reg_model.pkl")


def test_specify_gradient_boosting_regressor_return_types():
    """Test, if the function returns a dictionary."""
    train_encoded = pd.read_stata(
        BLD / "python" / "Lesson_A" / "data" / "train_encoded.dta"
    )
    model_features, reg_model = specify_gradient_boosting_regressor(
        train_encoded, 400, 4, 10, 0.01, "squared_error"
    )
    assert isinstance(model_features, pd.DataFrame)


def test_predict_net_value_train_r2_float():
    test = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "test.dta")
    reg_model = pickle.load(
        open(BLD / "python" / "Lesson_A" / "models" / "reg_model.pkl", "rb")
    )
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    regions_to_net = pickle.load(
        open(BLD / "python" / "Lesson_A" / "data" / "regions_to_net.pkl", "rb")
    )
    model_features = pd.read_stata(
        BLD / "python" / "Lesson_A" / "data" / "model_features.dta"
    )
    model_policy, train_r2, test_r2 = predict_net_value(
        train, test, reg_model, regions_to_net, model_features
    )
    assert isinstance(train_r2, float)


def test_predict_net_value_test_r2_float():
    test = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "test.dta")
    reg_model = pickle.load(
        open(BLD / "python" / "Lesson_A" / "models" / "reg_model.pkl", "rb")
    )
    train = pd.read_stata(BLD / "python" / "Lesson_A" / "data" / "train.dta")
    regions_to_net = pickle.load(
        open(BLD / "python" / "Lesson_A" / "data" / "regions_to_net.pkl", "rb")
    )
    model_features = pd.read_stata(
        BLD / "python" / "Lesson_A" / "data" / "model_features.dta"
    )
    model_policy, train_r2, test_r2 = predict_net_value(
        train, test, reg_model, regions_to_net, model_features
    )
    assert isinstance(test_r2, float)
