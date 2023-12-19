#Task file for C_Neural_networks

from pathlib import Path
from pytask import task
import pandas as pd
import tensorflow as tf
import keras
import pickle


from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.C_Neural_networks import preprocess_data, scale_datasets, build_model_using_sequential, compile_and_train_model, plot_history, predict_house_value, assess_performance



def task_preprocess_data(
    depends_on={"train_data": SRC / "data" / "california_housing_train.csv",
                "test_data": SRC / "data" / "california_housing_train.csv",
    },
    produces={
        "x_train": BLD / "python" / "Lesson_C" / "data" / "x_train.csv",
        "y_train": BLD / "python" / "Lesson_C" / "data" / "y_train.csv",
        "x_test": BLD / "python" / "Lesson_C" / "data" / "x_test.csv",
        "y_test": BLD / "python" / "Lesson_C" / "data" / "y_test.csv",
    },
):
    """Preprocess data."""
    train_data = pd.read_csv(depends_on["train_data"])
    test_data = pd.read_csv(depends_on["test_data"])
    x_train, y_train, x_test, y_test = preprocess_data(train_data, test_data)
    x_train.to_csv(produces["x_train"])
    y_train.to_csv(produces["y_train"])
    x_test.to_csv(produces["x_test"])
    y_test.to_csv(produces["y_test"])



def task_scale_datasets(
    depends_on={
        "x_train": BLD / "python" / "Lesson_C" / "data" / "x_train.csv",
        "x_test": BLD / "python" / "Lesson_C" / "data" / "x_test.csv",
    },
    produces={
        "x_train_scaled": BLD / "python" / "Lesson_C" / "data" / "x_train_scaled.csv",
        "x_test_scaled": BLD / "python" / "Lesson_C" / "data" / "x_test_scaled.csv",
    }
):
    """Scale datasets."""
    x_train = pd.read_csv(depends_on["x_train"])
    x_test = pd.read_csv(depends_on["x_test"])
    x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
    x_train_scaled.to_csv(produces["x_train_scaled"])
    x_test_scaled.to_csv(produces["x_test_scaled"])
    
    
    
def task_build_model_using_sequential(
    produces={
        "model": BLD / "python" / "Lesson_C" / "model" / "model.keras",
    }
):
    model = build_model_using_sequential()
    model.save(produces["model"])
    

def task_compile_and_train_model(
    depends_on={
        "model": BLD / "python" / "Lesson_C" / "model" / "model.keras",
        "x_train_scaled": BLD / "python" / "Lesson_C" / "data" / "x_train_scaled.csv",
        "y_train": BLD / "python" / "Lesson_C" / "data" / "y_train.csv",
    },
    produces={
        "history": BLD / "python" / "Lesson_C" / "model" / "history.pkl",
    }
):
    model = keras.models.load_model(depends_on["model"])
    x_train_scaled = pd.read_csv(depends_on["x_train_scaled"])
    y_train = pd.read_csv(depends_on["y_train"])
    history = compile_and_train_model(model, x_train_scaled, y_train)
    # Save history using pickle
    with open(produces["history"], "wb") as file:
        pickle.dump(history, file)


def task_plot_history(
    depends_on={
        "history": BLD / "python" / "Lesson_C" / "model" / "history.pkl",
    },
    produces={
        "plot": BLD / "python" / "Lesson_C" / "figures" / "history.png",
    }
):
    # Load history using pickle
    with open(depends_on["history"], "rb") as file:
        history = pickle.load(file)
    # Plot the history
    plot = plot_history(history, 'mean_squared_logarithmic_error')
    plot.savefig(produces["plot"])
    

def task_predict_house_value(
    depends_on={
        "model": BLD / "python" / "Lesson_C" / "model" / "model.keras",
        "x_test_scaled": BLD / "python" / "Lesson_C" / "data" / "x_test_scaled.csv",
        "x_test": BLD / "python" / "Lesson_C" / "data" / "x_test.csv",
    },
    produces={
        "x_test_pred": BLD / "python" / "Lesson_C" / "data" / "x_test_pred.csv",
    }
):
    model = keras.models.load_model(depends_on["model"])
    x_test_scaled = pd.read_csv(depends_on["x_test_scaled"])
    x_test = pd.read_csv(depends_on["x_test"])
    x_test_pred = predict_house_value(model, x_test_scaled, x_test)
    x_test_pred.to_csv(produces["x_test_pred"])

def task_assess_performance(
    depends_on={
        "x_test_pred": BLD / "python" / "Lesson_C" / "data" / "x_test_pred.csv",
        "y_test": BLD / "python" / "Lesson_C" / "data" / "y_test.csv",
    },
    produces={
        "rmse": BLD / "python" / "Lesson_C" / "model_fit" / "Rmse.txt"
    }
): 
    x_test_pred = pd.read_csv(depends_on["x_test_pred"])
    y_test = pd.read_csv(depends_on["y_test"])
    rmse = assess_performance(x_test_pred, y_test)
    with open(produces["rmse"], "w") as file:
        file.write(str(rmse))