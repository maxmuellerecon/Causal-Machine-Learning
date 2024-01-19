# A_Machine learning basics

# Import libraries
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

style.use("ggplot")
pd.options.mode.chained_assignment = None

from causalmachinelearning.lessons.__exceptions import (
    _fail_if_not_dataframe,
    _fail_if_too_many_quantiles,
    _fail_if_not_dict,
    _fail_if_irregular_learning_rate,
    _fail_if_not_gradient_boost,
)


###################A.1 Cross Validation##############################################
# Separate the good customers from the bad customers
def separate_customers(raw):
    """Sum up transactions for each customer.

    Args:
        raw (data frame): Passes the raw dataset

    Returns:
        DataFrame: Gives out the final dataframe

    """
    _fail_if_not_dataframe(raw)

    profitable = raw[["customer_id"]].assign(
        net_value=raw.drop(columns="customer_id").sum(axis=1)
    )
    return profitable


# Merge the customer features with the profitable dataset
def merge_customers(raw, merge_data):
    """Merge the customer features with the profitable dataset.

    Args:
        raw (data frame): Passes the raw dataset
        merge_data (data frame): Passes the customer features dataset

    Returns:
        DataFrame: Gives out the final merged dataframe

    """
    _fail_if_not_dataframe(raw)
    _fail_if_not_dataframe(merge_data)

    customer_features_merged = merge_data.merge(raw, on="customer_id")
    return customer_features_merged


# Split the data into train and test
def split_data(raw, size):
    """Split the data into train and test."""
    _fail_if_not_dataframe(raw)

    # Remove the variable level_0
    raw = raw.drop(columns="level_0")
    train, test = train_test_split(raw, test_size=size, random_state=13)
    train.shape, test.shape
    return train, test


# Plot net value by income quantiles
def plot_profit_by_income(raw, quantiles):
    """Plot net value by income quantiles.

    Args:
        raw (DataFrame): The data for profitablity and income
        quantiles (int): number of quantiles to be created

    Returns:
        plot: Plot of the net value by income quantiles

    """
    _fail_if_not_dataframe(raw)
    _fail_if_too_many_quantiles(quantiles)

    plt.figure(figsize=(12, 6))
    np.random.seed(
        123
    )  ## seed because the CIs from seaborn uses boostraping for inference
    sns.barplot(
        data=raw.assign(
            income_quantile=pd.qcut(raw["income"], q=quantiles)
        ),  # pd.qcut create quantiles of a column
        x="income_quantile",
        y="net_value",
    )
    plt.title("Profitability by Income")
    plt.xticks(rotation=70)
    return plt


def plot_profit_by_region(raw):
    """Plot net value by region."""
    _fail_if_not_dataframe(raw)

    plt.figure(figsize=(12, 6))
    np.random.seed(123)
    sns.barplot(data=raw, x="region", y="net_value")
    plt.title("Profitability by Region")
    return plt


# Filter regions, that are significantly profitable
# extract the lower bound of the 95% CI from the plot above
def lower_bound_CI(raw):
    """Filter regions, that are significantly profitable
    Args:
        raw (DataFrame): The data for profitablity and income

    Returns:
        dict: Dictionary of regions and their lower bound of the 95% CI
    """
    _fail_if_not_dataframe(raw)

    regions_to_net = raw.groupby("region")["net_value"].agg(["mean", "count", "std"])
    regions_to_net = regions_to_net.assign(
        lower_bound=regions_to_net["mean"]
        - 1.96 * regions_to_net["std"] / (regions_to_net["count"] ** 0.5)
    )
    regions_to_net_lower_bound = regions_to_net["lower_bound"].to_dict()
    regions_to_net = regions_to_net["mean"].to_dict()
    # filters regions where the net value lower bound is > 0.
    regions_to_invest = {
        region: net for region, net in regions_to_net_lower_bound.items() if net > 0
    }
    return regions_to_net, regions_to_invest


# Check out of sample performance
def plot_oos_performance(rawtest, regions):
    """Filter regions in test dataset, that are profitable in the training data and plot
    the average net income in test data.

    Args:
        regions (Dict): Dictionary describing the keys to the regions that are profitable in the training data

    Returns:
        plot: Plot of the average net income in test data

    """
    _fail_if_not_dataframe(rawtest)
    _fail_if_not_dict(regions)

    region_policy = rawtest[
        rawtest["region"].isin(regions.keys())
    ]  # filter regions in regions_to_invest
    sns.histplot(data=region_policy, x="net_value")
    # average has to be over all customers, not just the one we've filtered with the policy
    plt.title(
        "Average Net Income: %.2f"
        % (region_policy["net_value"].sum() / rawtest.shape[0])
    )
    return plt


############################A.2 Motivating Example: Gradient Boosting Regressor#############################
# Encode the region variable
def encode_region(df, regions_to_net):
    """"Encode regions with dictionary, check if "level_0" exists before trying to drop
    it."""
    _fail_if_not_dataframe(df)

    if "level_0" in df.columns:
        df = df.drop(columns="level_0")
    df["region"] = df["region"].map(regions_to_net)
    return df


def specify_gradient_boosting_regressor(
    df, n_estimators, max_depth, min_samples_split, learning_rate, loss
):
    """Run Gradient Boosting Regressor on the training data and predict the net value on
    the test data.

    Args:
        df (DataFrame): Training data
        n_estimators (int): The number of boosting stages to perform
        max_depth (int): Maximum depth of the individual regression estimators
        min_samples_split (int): The minimum number of samples required to split an internal node
        learning_rate (float): Learning rate shrinks the contribution of each tree by learning_rate
        loss (str): loss function to be optimized

    Returns:
        float: R2 score of the model

    """
    _fail_if_not_dataframe(df)
    _fail_if_irregular_learning_rate(learning_rate)

    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "learning_rate": learning_rate,
        "loss": loss,
    }
    features = ["region", "income", "age"]
    target = "net_value"
    np.random.seed(123)
    reg = ensemble.GradientBoostingRegressor(**model_params)
    # fit model on the training set
    model_features = df[features]
    reg.fit(model_features, df[target])
    return model_features, reg


# Predict the net value
def predict_net_value(raw_train, raw_test, reg, regions_to_net, model_features):
    """Predict the net value on the test data and calculate the R2 score."""
    _fail_if_not_dataframe(raw_train)
    _fail_if_not_dataframe(raw_test)
    _fail_if_not_dataframe(model_features)
    _fail_if_not_gradient_boost(reg)

    features = ["region", "income", "age"]
    target = "net_value"
    train_pred = model_features.assign(
        predictions=reg.predict(model_features[features])
    )
    # Print the R2 score, for the testing data, we need to replace regions integers with regions_to_net (pipe)
    Train_R2 = r2_score(y_true=raw_train[target], y_pred=train_pred["predictions"])
    Test_R2 = r2_score(
        y_true=raw_test[target],
        y_pred=reg.predict(raw_test[features].pipe(encode_region, regions_to_net)),
    )
    # Assign predictions to test data
    test = encode_region(raw_test, regions_to_net)
    model_policy = test.assign(predictions=reg.predict(test[features]))
    return model_policy, Train_R2, Test_R2
