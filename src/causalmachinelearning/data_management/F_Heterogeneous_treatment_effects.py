#F_Heterogeneous Treatment effects 

#Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


############################F.1 Linear regression analysis##############################################
#Goal: Predict when to charge more for ice cream, depending on temperature, weekday and cost
#Split data into train and test


def split_data(data):
    """Run linear regression models

    Args:
        data (data): data frame

    Returns:
        train: training data
        test: testing data
    """
    np.random.seed(123)
    train, test = train_test_split(data, test_size=0.3)
    return train, test
    
    
    
def regressions_three(train):
    """Run linear regression models

    Args:
        train (data): training data
        
    Returns:
        m1 (LinearRegression): model to be used
        m2 (LinearRegression): model to be used
        m3 (LinearRegression): model to be used
        latex_code1 (str): latex code for table 1
        latex_code2 (str): latex code for table 2
        latex_code3 (str): latex code for table 3
    """
    #Linear regression
    m1 = smf.ols("sales ~ price + temp+C(weekday)+cost", data=train).fit() #Include the C to make it categorical
    table1 = m1.summary().tables[1]
    print(table1)
    # Export to LaTeX
    latex_code1 = table1.as_latex_tabular()
    print(latex_code1)
    
    #Linear regression with one interaction
    m2 = smf.ols("sales ~ price*temp + C(weekday) +cost", data=train).fit()
    table2 = m2.summary().tables[1]
    print(table2)
    # Export to LaTeX
    latex_code2 = table2.as_latex_tabular()
    print(latex_code2)

    #Linear regression with all interactions
    m3 = smf.ols("sales ~ price*temp + price*C(weekday) + price*cost", data=train).fit()
    table3 = m3.summary().tables[1]
    print(table3)
    # Export to LaTeX
    latex_code3 = table3.as_latex_tabular()
    print(latex_code3)
    return m1, m2, m3, latex_code1, latex_code2, latex_code3




###########################F.2 Sensitivity analysis#####################################################
#Predict the effect of a change in price by 1 on sales
def pred_sensitivity(m, df, t="price"):
    """Predict the effect of a change in price by 1 on sales

    Args:
        m (LinearRegression): model to be used
        df (data): testing data on which to predict
        t (str, optional): Variable to predict. Defaults to "price".

    Returns:
        DataFrame: Data frame with predicted sensitivity
    """
    return df.assign(**{
        "pred_sens": m.predict(df.assign(**{t:df[t]+1})) - m.predict(df)
    })



# ##########################F.3 Comparing CATE model to ML prediction model###############################

#ML Model using price, temp, weekday and cost to predict sales
def ml_model(train, test):
    """ML Model using price, temp, weekday and cost to predict sales

    Args:
        train (data): training data
        test (data): testing data

    Returns:
        ml (ML): ML model
    """
    X = ["price", "temp", "weekday", "cost"]
    Y = ["sales"]
    m4 = GradientBoostingRegressor()
    m4.fit(train[X], train[Y])
    #Check that model does not overfit
    m4.score(test[X], test[Y])
    return m4


#Segment the units into 2 groups based on the sensitivity predictions
def comparison(regr_data, ml_model, groups):
    """Predict bands in regression and ml model

    Args:
        regr_model (DataFrame): Data fromRegression model
        ml_model (model): ML model
        groups (int): number of bands

    Returns:
        bands_df (DataFrame): Data frame with predicted sensitivity
    """
    X = ["price", "temp", "weekday", "cost"]
    bands_df = regr_data.assign(
        sens_band = pd.qcut(regr_data["pred_sens"], groups), # create two groups based on sensitivity predictions 
        pred_sales = ml_model.predict(regr_data[X]),
        pred_band = pd.qcut(ml_model.predict(regr_data[X]), groups), # create two groups based on sales predictions
    )
    return bands_df


def plot_regr_model(data):
    """Plot the regression model - second partition, sales are not as sensitive to price changes as for the first partition"""
    g = sns.FacetGrid(data, col="sens_band")
    g.map_dataframe(sns.regplot, x="price", y="sales")
    g.set_titles(col_template="Sens. Band {col_name}")
    plt.show() 
    return plt


def plot_ml_model(data):
    """Plot the ML model - Here elasticity is about the same for both groups"""
    g = sns.FacetGrid(data, col="pred_band")
    g.map_dataframe(sns.regplot, x="price", y="sales")
    g.set_titles(col_template="Pred. Band {col_name}");
    plt.show() 
    return plt
