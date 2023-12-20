#D_Decision_Trees.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.tree import export_graphviz
import graphviz




####################D. Decision Trees for Regression##############################################	
#Supervised learning algorithm used for regression 
#Positive: Non-parametric, can approximate any function, can be used for classification and regression
#Negative: Can only handle structured data (supervised learning)


def create_data():
    dataset = np.array( 
    [['Asset Flip', 100, 1000], 
    ['Text Based', 500, 3000], 
    ['Visual Novel', 1500, 5000], 
    ['2D Pixel Art', 3500, 8000], 
    ['2D Vector Art', 5000, 6500], 
    ['Strategy', 6000, 7000], 
    ['First Person Shooter', 8000, 15000], 
    ['Simulator', 9500, 20000], 
    ['Racing', 12000, 21000], 
    ['RPG', 14000, 25000], 
    ['Sandbox', 15500, 27000], 
    ['Open-World', 16500, 30000], 
    ['MMOFPS', 25000, 52000], 
    ['MMORPG', 30000, 80000] 
    ])
    return dataset

def preprocess_data(dataset):
    # Convert non-numeric columns to numeric
    for i in range(dataset.shape[1]):
        try:
            dataset[:, i] = dataset[:, i].astype(int)
        except ValueError:
            # Handle non-numeric values (e.g., 'Asset Flip') by assigning them a default value (0 in this case)
            dataset[:, i] = 0
    return dataset


def fit_tree(dataset):
    dataset = np.array(dataset)
    dataset = preprocess_data(dataset)
    X = dataset[:, 1].reshape(-1, 1)
    y = dataset[:, 2]
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state=0)
    # fit the regressor with X and Y data
    regressor.fit(X, y)
    return regressor


def predict_new_value(regressor, new_value):
    y_pred = regressor.predict([[new_value]])
    print("Predicted price: % d\n"% y_pred)  
    return y_pred


def visualize_decision_tree(regressor, feature_names):
    dot_data = export_graphviz(regressor, out_file=None, 
                               feature_names=feature_names,
                               filled=True, rounded=True, special_characters=True)
    
    graph = graphviz.Source(dot_data)
    return graph