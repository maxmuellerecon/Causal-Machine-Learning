#Task for for E_Ensemble_learning


from pathlib import Path
from pytask import task
import pandas as pd


from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management._Backup_E_Ensemble_learning import create_data, split_train_test, train_and_predict


def task_create_data(
    produces={
        "data": BLD / "python" / "Lesson_E" / "data" / "gym_data.csv",
    }
):
    """Create data."""
    data = create_data()
    data.to_csv(produces["data"])
    
    
def task_split_train_test(
    depends_on={
        "data": BLD / "python" / "Lesson_E" / "data" / "gym_data.csv",
    },
    produces={
        "X_train": BLD / "python" / "Lesson_E" / "data" / "X_train.csv",
        "X_test": BLD / "python" / "Lesson_E" / "data" / "X_test.csv",
        "y_train": BLD / "python" / "Lesson_E" / "data" / "y_train.csv",
        "y_test": BLD / "python" / "Lesson_E" / "data" / "y_test.csv",
    }
):
    """Split data into train and test."""
    data = pd.read_csv(depends_on["data"])
    X_train, X_test, y_train, y_test = split_train_test(data)
    X_train.to_csv(produces["X_train"])
    X_test.to_csv(produces["X_test"])
    y_train.to_csv(produces["y_train"])
    y_test.to_csv(produces["y_test"])
    
    
def task_train_and_predict(
    depends_on={
        "X_train": BLD / "python" / "Lesson_E" / "data" / "X_train.csv",
        "X_test": BLD / "python" / "Lesson_E" / "data" / "X_test.csv",
        "y_train": BLD / "python" / "Lesson_E" / "data" / "y_train.csv",
    },
    produces={
        "y_test_pred": BLD / "python" / "Lesson_E" / "data" / "y_test_pred.csv",
    }
):
    """Train and predict using Random Forests."""
    X_train = pd.read_csv(depends_on["X_train"])
    X_test = pd.read_csv(depends_on["X_test"])
    y_train = pd.read_csv(depends_on["y_train"])
    y_test_pred = train_and_predict(X_train, X_test, y_train)
    y_test_pred.to_csv(produces["y_test_pred"]) 