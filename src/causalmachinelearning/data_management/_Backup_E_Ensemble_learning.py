# '#E_Ensemble_learning


# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn import metrics

# ####################E.1 Bagging: Random Forests##############################################
# #As an example for a bagging technique, we will use Random Forests
# #Bagging: bootstrap aggregation, where we take random samples of data with replacement and train a decision tree on each sample
# #Positive: Every decision tree has high variance, but when we combine all of them in parallel then the resultant variance is low


# def create_data():
#     """Create data and plot it"""
#     ColumnNames = ["Hours", 'Calories', 'Weight']
#     DataValues = [[1.0, 2500, 95],
#                   [2.0, 2000, 85],
#                   [2.5, 1900, 83],
#                   [3.0, 1850, 81],
#                   [3.5, 1600, 80],
#                   [4.0, 1500, 78],
#                   [5.0, 1500, 77],
#                   [5.5, 1600, 80],
#                   [6.0, 1700, 75],
#                   [6.5, 1500, 70]]
#     # Create the Data Frame
#     GymData = pd.DataFrame(data=DataValues, columns=ColumnNames)
#     return GymData


# def split_train_test(GymData):
#     # Separate Target Variable and Predictor Variables
#     TargetVariable = 'Weight'
#     Predictors = ['Hours', 'Calories']
#     X = GymData[Predictors].values
#     y = GymData[TargetVariable].values
#     # Split the data into training and testing set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train = pd.DataFrame(data=X_train, columns=Predictors)
#     X_test = pd.DataFrame(data=X_test, columns=Predictors)
#     y_train = pd.DataFrame(data=y_train, columns=[TargetVariable])
#     y_test = pd.DataFrame(data=y_test, columns=[TargetVariable])
#     return X_train, X_test, y_train, y_test


# def train_and_predict(X_train, X_test, y_train):
#     """Train and predict using Random Forests"""
#     # Create the model object
#     print("X_train shape:", X_train.shape)
#     print("y_train shape:", y_train.shape)
#     print("X_test shape:", X_test.shape)
#     RegModel = RandomForestRegressor(n_estimators=100, criterion='squared_error')  # Use 'mse' for mean squared error
#     # Creating the model on Training Data
#     RF = RegModel.fit(X_train, y_train.values.ravel())
#     prediction = RF.predict(X_test).ravel()
#     y_test_pred = pd.DataFrame(data={'Predicted Weight': prediction})
#     return y_test_pred
    
    

# # print('R2 Value:',metrics.r2_score(y_train, RF.predict(X_train)))
 
# # #Measuring accuracy on Testing Data
# # print('Accuracy',100- (np.mean(np.abs((y_test - prediction) / y_test)) * 100))'