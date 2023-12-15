#tasks for 01_Machine Learning basics  

from pathlib import Path
from pytask import task
import pandas as pd
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.A_Machine_learning_basics import separate_customers, merge_customers, split_data, plot_profit_by_income, plot_profit_by_region, lower_bound_CI, plot_oos_performance, encode_region


def task_separate_customers(
    depends_on=SRC / "data" / "customer_transactions.csv",
    produces=BLD / "python" / "data" / "profitable.dta",
):
    """Separate customers."""
    customer_transactions = pd.read_csv(depends_on)
    profitable = separate_customers(customer_transactions)
    profitable.to_stata(produces)



merge_data_deps = {
    "profitable": BLD / "python" / "data" / "profitable.dta",
    "customer_features": SRC / "data" / "customer_features.csv",
}
def task_merge_customers(
    depends_on=merge_data_deps,
    produces=BLD / "python" / "data" / "customer_features_merged.dta",
):
    """Merge customers."""
    profitable = pd.read_stata(depends_on["profitable"])
    customer_features = pd.read_csv(depends_on["customer_features"])
    customer_features_merged = merge_customers(profitable, customer_features)
    customer_features_merged.to_stata(produces)



split_data_out = {
    "train": BLD / "python" / "data" / "train.dta",
    "test": BLD / "python" / "data" / "test.dta",
}
def task_split_data(
    depends_on=BLD / "python" / "data" / "customer_features_merged.dta",
    produces=split_data_out,
):
    """Split data into test and training."""
    customer_features_merged = pd.read_stata(depends_on)
    train, test = split_data(customer_features_merged, 0.3)
    train.to_stata(produces["train"])
    test.to_stata(produces["test"])
    

def task_plot_profit_by_income(
    depends_on= BLD / "python" / "data" / "train.dta",
    produces=BLD / "python" / "figures" / "profit_by_income.png",
):
    """Plot profit by income."""
    train = pd.read_stata(depends_on)
    fig = plot_profit_by_income(train, 10)
    fig.savefig(produces)


def task_plot_profit_by_region(
    depends_on= BLD / "python" / "data" / "train.dta",
    produces=BLD / "python" / "figures" / "profit_by_region_plot.png",
):
    """Plot profit by region."""
    train = pd.read_stata(depends_on)
    fig = plot_profit_by_region(train)
    fig.savefig(produces)
    

lower_bound_CI_out = {
    "regions_to_net": BLD / "python" / "data" / "regions_to_net.pkl",
    "regions_to_invest": BLD / "python" / "data" / "regions_to_invest.pkl",
}
def task_lower_bound_CI(
    depends_on= BLD / "python" / "data" / "train.dta",
    produces=lower_bound_CI_out,
):
    """Filter regions, that are significantly profitable."""
    train = pd.read_stata(depends_on)
    regions_to_net, regions_to_invest = lower_bound_CI(train)
    with open(produces["regions_to_net"], "wb") as f:
        pickle.dump(regions_to_net, f)
    with open(produces["regions_to_invest"], "wb") as f:
        pickle.dump(regions_to_invest, f)


def task_plot_oos_performance(
    depends_on={
        "test": BLD / "python" / "data" / "test.dta",
        "regions_to_invest": BLD / "python" / "data" / "regions_to_invest.pkl"
        },
    produces= BLD / "python" / "figures" / "oos_performance.png",
):
    """Plot out of sample performance."""
    test = pd.read_stata(depends_on["test"])
    with open(depends_on["regions_to_invest"], "rb") as f:
        regions_to_invest = pickle.load(f)
    plot = plot_oos_performance(test, regions_to_invest)
    plot.savefig(produces)


def task_encode_region(
    depends_on={
        "train": BLD / "python" / "data" / "train.dta",
        "regions_to_net": BLD / "python" / "data" / "regions_to_net.pkl"
        },
    produces= BLD / "python" / "data" / "train_encoded.dta",
):
    """Plot out of sample performance."""
    train = pd.read_stata(depends_on["train"])
    with open(depends_on["regions_to_net"], "rb") as f:
        regions_to_net = pickle.load(f)
    train_encoded = encode_region(train, regions_to_net)
    train_encoded.to_stata(produces)
