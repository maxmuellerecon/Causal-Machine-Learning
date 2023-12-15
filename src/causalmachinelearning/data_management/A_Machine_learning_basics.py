#A_Machine learning basics

#Import libraries 
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier 
import seaborn as sns
import os
from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")

###################A.1 Cross Validation##############################################

#Separate the good customers from the bad customers
def separate_customers(raw):
    """Sum up transactions for each customer
    
    Args:
        raw (data frame): Passes the raw dataset

    Returns:
        DataFrame: Gives out the final dataframe
    """
    profitable = (raw[["customer_id"]]
              .assign(net_value = raw
                      .drop(columns="customer_id")
                      .sum(axis=1)))
    return profitable

#Merge the customer features with the profitable dataset
def merge_customers(raw, merge_data):
    """Merge the customer features with the profitable dataset"""
    customer_features_merged = (merge_data
                        .merge(raw, on="customer_id"))
    return customer_features_merged


#Split the data into train and test
def split_data(raw, size):
    """Split the data into train and test"""
    #Remove the variable level_0
    raw = raw.drop(columns="level_0")
    train, test = train_test_split(raw, test_size=size, random_state=13)
    train.shape, test.shape
    return train, test


#Plot net value by income quantiles
def plot_profit_by_income(raw, quantiles):
    """Plot net value by income quantiles

    Args:
        raw (DataFrame): The data for profitablity and income
        quantiles (int): number of quantiles to be created
    """
    plt.figure(figsize=(12,6))
    np.random.seed(123) ## seed because the CIs from seaborn uses boostraping for inference
    sns.barplot(data=raw.assign(income_quantile=pd.qcut(raw["income"], q=quantiles)),  # pd.qcut create quantiles of a column
            x="income_quantile", y="net_value")
    plt.title("Profitability by Income")
    plt.xticks(rotation=70)
    return plt


def plot_profit_by_region(raw):
    """Plot net value by region"""
    plt.figure(figsize=(12,6))
    np.random.seed(123)
    sns.barplot(data=raw, x="region", y="net_value")
    plt.title("Profitability by Region")
    return plt


#Filter regions, that are significantly profitable
# extract the lower bound of the 95% CI from the plot above
def lower_bound_CI(raw):
    """Filter regions, that are significantly profitable
    Args:
        raw (DataFrame): The data for profitablity and income
    
    Returns:
        dict: Dictionary of regions and their lower bound of the 95% CI
    """    
    regions_to_net = raw.groupby('region')['net_value'].agg(['mean', 'count', 'std'])
    regions_to_net = regions_to_net.assign(
    lower_bound=regions_to_net['mean'] - 1.96*regions_to_net['std']/(regions_to_net['count']**0.5)
    )
    regions_to_net_lower_bound = regions_to_net['lower_bound'].to_dict()
    regions_to_net = regions_to_net['mean'].to_dict()
    # filters regions where the net value lower bound is > 0.
    regions_to_invest = {region: net 
                     for region, net in regions_to_net_lower_bound.items()
                     if net > 0}
    return regions_to_net, regions_to_invest



#Check out of sample performance
def plot_oos_performance(rawtest, regions):
    """Filter regions in test dataset, that are profitable in the training data and plot the average net income in test data

    Args:
        regions (Dict): Dictionary describing the keys to the regions that are profitable in the training data
    
    Returns:
        plot: Plot of the average net income in test data
    """
    region_policy = (rawtest[rawtest["region"].isin(regions.keys())]) # filter regions in regions_to_invest                    
    sns.histplot(data=region_policy, x="net_value")
    # average has to be over all customers, not just the one we've filtered with the policy
    plt.title("Average Net Income: %.2f" % (region_policy["net_value"].sum() / rawtest.shape[0]));
    return plt



# ###########################A.2 Gradient Boosting Regressor#############################


#Encode the region variable
def encode_region(df, regions_to_net):
    df = df.drop(columns="level_0")
    df['region'] = df['region'].map(regions_to_net)
    return df

# def specify_gradient_boosting_regressor(df, n_estimators, max_depth, min_samples_split, learning_rate, loss):
#     """Run Gradient Boosting Regressor on the training data and predict the net value on the test data

#     Args:
#         df (DataFrame): Training data
#         n_estimators (int): The number of boosting stages to perform
#         max_depth (int): Maximum depth of the individual regression estimators
#         min_samples_split (int): The minimum number of samples required to split an internal node
#         learning_rate (float): Learning rate shrinks the contribution of each tree by learning_rate
#         loss (str): loss function to be optimized
    
#     Returns:
#         float: R2 score of the model
#     """
#     model_params = {'n_estimators': n_estimators,
#                     'max_depth': max_depth,
#                     'min_samples_split': min_samples_split,
#                     'learning_rate': learning_rate,
#                     'loss': loss}
#     features = ["region", "income", "age"]
#     target = "net_value"
#     np.random.seed(123)
#     reg = ensemble.GradientBoostingRegressor(**model_params)
#     # fit model on the training set
#     encoded_train = df[features]
#     reg.fit(encoded_train, df[target]);
#     return features, target, encoded_train, reg

# #Train the model
# features, target, encoded_train, reg = specify_gradient_boosting_regressor(train_encoded, 400, 4, 10, 0.01, 'squared_error')


# #Predict the net value
# train_pred = (encoded_train.assign(predictions=reg.predict(encoded_train[features])))
# #Print the R2 score, for the testing data, we need to replace regions integers with regions_to_net (pipe)
# print("Train R2: ", r2_score(y_true=train[target], y_pred=train_pred["predictions"]))
# print("Test R2: ", r2_score(y_true=test[target], y_pred=reg.predict(test[features].pipe(encode_region))))

# #Assign predictions to test data
# test = encode_region(test)
# model_policy = test.assign(predictions=reg.predict(test[features]))

# #Plot the predictions
# def plot_prediction_quantiles(raw, n_bands):
#     """Plot the predictions"""
#     plt.figure(figsize=(12,6))   
#     bands = [f"band_{b}" for b in range(1,n_bands+1)]
#     np.random.seed(123)
#     model_plot = sns.barplot(data=raw.assign(model_band = pd.qcut(raw["predictions"], q=n_bands)),       #pd.qcut create quantiles of a column, here from predictions
#                          x="model_band", y="net_value")
#     plt.title("Profitability by Model Prediction Quantiles")
#     plt.xticks(rotation=70)
#     plt.savefig(os.path.join(BLDFIG, "01_4_pred_quantiles.png"))
#     return plt.show()

# plot_prediction_quantiles(model_policy, 50)

# #Plot the comparison between regional and model policy
# def plot_comparison(raw):
#     """Plot the comparison between regional and model policy"""
#     plt.figure(figsize=(10,6))
#     model_plot_df = (raw[raw["predictions"]>0])
#     sns.histplot(data=model_plot_df, x="net_value", color="C2", label="model_policy")
#     raw['region'] = raw['region'].map({v: k for k, v in regions_to_net.items()})            #Replace the values in the region column with the keys from regions_to_net
#     region_plot_df = (raw[raw["region"].isin(regions_to_invest.keys())])
#     sns.histplot(data=region_plot_df, x="net_value", color="C1", label="region_policy")
#     plt.title("Model Net Income: %.2f;    Region Policy Net Income %.2f." % 
#           (model_plot_df["net_value"].sum() / test.shape[0],
#            region_plot_df["net_value"].sum() / test.shape[0]))
#     plt.legend();
#     plt.savefig(os.path.join(BLDFIG, "01_5_comparison.png"))
#     return plt.show()

# plot_comparison(model_policy)
# #The model policy is better than the regional policy, but only marginally

# #Thresholding, ergo checking out the bins
# def model_binner(prediction_column, bins):
#     """Swap binary outcome(predictions > 0) with a continuous decision along the bins)"""
#     # find the bins according to the training set
#     bands = pd.qcut(prediction_column, q=bins, retbins=True)[1]
#     def binner_function(prediction_column):
#         return np.digitize(prediction_column, bands)
#     return binner_function
    
# # train the binning function
# binner_fn = model_binner(train_pred["predictions"], 20)

# # apply the binning, most profitable bins are 19 and 20, from extensive to intensive margin, binary decision to continuous decision
# model_band = model_policy.assign(bands = binner_fn(model_policy["predictions"]))
# model_band.head()