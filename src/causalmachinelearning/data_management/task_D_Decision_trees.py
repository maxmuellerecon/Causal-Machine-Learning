#Task file for D_Decision_trees


from pathlib import Path
from pytask import task
import pandas as pd
import keras
import pickle


from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.D_Decision_trees import create_data, fit_tree, predict_new_value, visualize_decision_tree


def task_create_data(
    produces={
        "dataset": BLD / "python" / "Lesson_D" / "data" / "dataset.csv",
    }
):
    """Create data."""
    dataset = create_data()
    data = pd.DataFrame(dataset)
    data.to_csv(produces["dataset"])
    
    
    
def task_fit_tree(
    depends_on={
        "dataset": BLD / "python" / "Lesson_D" / "data" / "dataset.csv",
    },
    produces={
        "regressor": BLD / "python" / "Lesson_D" / "model" / "regressor.pkl",
    }
):
    """Fit tree."""
    dataset = pd.read_csv(depends_on["dataset"])
    regressor = fit_tree(dataset)
    pickle.dump(regressor, open(produces["regressor"], 'wb'))
    


def task_predict_new_value(
    depends_on={
        "regressor": BLD / "python" / "Lesson_D" / "model" / "regressor.pkl",
    },
    produces={
        "y_pred": BLD / "python" / "Lesson_D" / "data" / "y_pred.csv",
    }
):
    """Predict new value."""
    regressor = pickle.load(open(depends_on["regressor"], 'rb'))
    new_value = 3750
    y_pred = predict_new_value(regressor, new_value)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv(produces["y_pred"])
    
    
    
def task_plot_tree(
    depends_on={
        "regressor": BLD / "python" / "Lesson_D" / "model" / "regressor.pkl",
    },
    produces={
        "plot": BLD / "python" / "Lesson_D" / "figures" / "tree.dot",
    }
):
    """Plot values."""
    regressor = pickle.load(open(depends_on["regressor"], 'rb'))
    graph = visualize_decision_tree(regressor, ['Production Cost'])
    graph.render(produces["plot"])