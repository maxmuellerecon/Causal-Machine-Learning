#C_Neural Networks

import math
import pandas as pd
from keras import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredLogarithmicError


####################C. Vanilla Neural Networks##############################################	
#Positive: Non-parametric, can approximate any function, can be used for classification and regression
#Negative: Can only handle structured data (supervised learning)

def preprocess_data(train_data, test_data):
    """Preprocess data, split target and predictors"""
    TARGET_NAME = 'median_house_value'
    x_train, y_train = train_data.drop(TARGET_NAME, axis=1), train_data[TARGET_NAME]
    x_test, y_test = test_data.drop(TARGET_NAME, axis=1), test_data[TARGET_NAME]
    return x_train, y_train, x_test, y_test


def scale_datasets(x_train, x_test):
  """ Standard Scale test and train data  Z - Score normalization (substract each data from its mean and divides it by the standard deviation of the data), get faster convergence to global optimum"""
  standard_scaler = StandardScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled


# Creating model using the Sequential in tensorflow-keras
def build_model_using_sequential():
    """Generate NN, First layer has 160 neurons, second layer has 480 neurons, third layer has 256 neurons, output layer has 1 neuron
    where, the 3 layers use relu activation function and the output layer uses linear activation function

    Returns:
        model: The model, that we have to train
    """
    hidden_units1 = 160
    hidden_units2 = 480
    hidden_units3 = 256
    model = Sequential([
    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
    Dense(1, kernel_initializer='normal', activation='linear')
    ])
    return model


# loss function
def compile_and_train_model(model, x_train_scaled, y_train):
    """Compile and train the model, We use Mean Squared Logarithmic Loss as loss function and metric, and Adam loss function optimizer.
    Args:
        model: The model, that we have to train
        x_train_scaled (DataFrame): Scaled train data
        y_train (DataFrame): Target train data
        
    Returns:
        history: The history of the training
    """
    learning_rate = 0.01
    msle = MeanSquaredLogarithmicError()
    model.compile(
        loss=msle, 
        optimizer=Adam(learning_rate=learning_rate), 
        metrics=[msle]
    )
    # train the model
    history = model.fit(
        x_train_scaled.values, 
        y_train.values, 
        epochs=10, 
        batch_size=64,
        validation_split=0.2
    )
    return history


def predict_house_value(model, x_test_scaled, x_test):
    """Predict house value using the model and safe it in the testing data"""
    x_test['prediction'] = model.predict(x_test_scaled)
    return x_test


def assess_performance(x_test_pred, y_test):
    """Assess the performance of the model, by calculating the root mean squared error"""
    rmse = math.sqrt(
        sum((x_test_pred['prediction'] - y_test["median_house_value"])**2) / len(x_test_pred)
    )
    return rmse