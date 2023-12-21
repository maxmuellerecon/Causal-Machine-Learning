#E_Ensemble Learning

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


####################E.1 Bagging ##############################################
#As an example for a bagging technique, we will use Random Forests
#Bagging: bootstrap aggregation, where we take random samples of data with replacement and train a decision tree on each sample
#Positive: Every decision tree has high variance, but when we combine all of them in parallel then the resultant variance is low

def make_dataset():
    """Make dataset for regression"""
    n_samples = 100 # Number of samples
    n_features = 6 # Number of features
    n_informative = 3 # Number of informative features i.e. actual features which influence the output
    X, y, coef = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                random_state=None, shuffle=False,noise=20,coef=True)
    df1 = pd.DataFrame(data=X,columns=['X'+str(i) for i in range(1,n_features+1)])
    df2=pd.DataFrame(data=y,columns=['y'])
    df=pd.concat([df1,df2],axis=1)
    return X, y, df, coef


def decision_tree(X, y):
    """Fit simple decision tree model and calculate R2 score
    Args:
        X np.array: features of model
        y np.array: target of model

    Returns:
        tree_model: the fitted model
        r2: R2 score
    """
    tree_model = tree.DecisionTreeRegressor(max_depth=5, random_state=None)
    tree_model.fit(X, y)
    # Calculate R2 score
    y_pred = tree_model.predict(X)
    r2 = r2_score(y, y_pred)
    print("R2 Score:", r2)
    return tree_model, r2

#R2_Train is 0.9811014350667862, so it is overfitting


def random_forest(X, y):
    """Fit forest model and calculate R2 score
    Args:
        X np.array: features of model
        y np.array: target of model

    Returns:
        forest_model: the fitted model
        r2: R2 score
    """
    forest_model = RandomForestRegressor(max_depth=5, random_state=None, max_leaf_nodes=5, n_estimators=100)
    forest_model.fit(X, y)
    y_pred = forest_model.predict(X)
    r2 = r2_score(y, y_pred)
    return forest_model, r2

#R2_Train is 0.8649247897434318, we prevent overfitting


####################E.2 Boosting ##############################################
#As an example for a boosting technique, we will use Gradient Boosting in random forests
#Boosting: sequential ensemble method, where we use errors of the previous weak lerner to train the next weak learner


def load_and_split_data():
    """Load and split data into train and test sets"""
    ch = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(ch.data, ch.target, random_state=42, test_size=0.1)
    # Create Training and Test Split
    X_train, X_test, y_train, y_test = train_test_split(ch.data, ch.target, random_state=42, test_size=0.1)
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    # Standardize the dataset
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std


def gradient_boosting_and_accuracy(X_train_std, y_train, X_test_std, y_test):
    # Hyperparameters for GradientBoostingRegressor
    gbr_params = {'n_estimators': 1000,
            'max_depth': 3,
            'min_samples_split': 5,
            'learning_rate': 0.01,
            'loss': 'squared_error'}
    # Create an instance of gradient boosting regressor
    gbr = GradientBoostingRegressor(**gbr_params)
    # Fit the model
    gbr.fit(X_train_std, y_train)
    # Print Coefficient of determination R^2
    r2 = gbr.score(X_test_std, y_test)
    # Create the mean squared error
    mse = mean_squared_error(y_test, gbr.predict(X_test_std))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    return gbr, mse, r2


def determine_feature_importance(gbr, X_test_std, y_test):
    # Get Feature importance data using feature_importances_ attribute
    ch = fetch_california_housing()
    feature_importance = gbr.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(8, 8))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(ch.feature_names)[sorted_idx])
    plt.title('Feature Importance (MDI)')
    result = permutation_importance(gbr, X_test_std, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig.tight_layout()
    return plt