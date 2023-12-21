#Task for for E_Ensemble_learning


from pathlib import Path
from pytask import task
import pandas as pd
import numpy as np
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.E_Ensemble_learning import make_dataset, decision_tree, random_forest, load_and_split_data, standardize_data, gradient_boosting_and_accuracy, determine_feature_importance


def task_make_dataset(
    produces={
        "X": BLD / "python" / "Lesson_E" / "data" / "X.npy",
        "y": BLD / "python" / "Lesson_E" / "data" / "y.npy",
        "df": BLD / "python" / "Lesson_E" / "data" / "df.csv",
        "coef": BLD / "python" / "Lesson_E" / "data" / "coef.npy",
    }
):
    """Make dataset."""
    X, y, df, coef = make_dataset()
    np.save(produces["X"], X)
    np.save(produces["y"], y)
    df.to_csv(produces["df"])
    np.save(produces["coef"], coef
)
    

def task_decision_tree(
    depends_on={
        "X": BLD / "python" / "Lesson_E" / "data" / "X.npy",
        "y": BLD / "python" / "Lesson_E" / "data" / "y.npy",
    },
    produces={
        "tree_model": BLD / "python" / "Lesson_E" / "model" / "tree_model.pkl",
        "R2_file": BLD / "python" / "Lesson_E" / "model_fit" / "R2_Tree.txt"
    }
):
    """Decision tree."""
    X = np.load(depends_on["X"])
    y = np.load(depends_on["y"])
    tree_model, r2 = decision_tree(X, y)
    with open(produces["tree_model"], "wb") as f:
        pickle.dump(tree_model, f)
    with open(produces["R2_file"], "w") as txt_file:
        txt_file.write(f"R2_Train: {r2}\n")


def task_random_forest(
    depends_on={
        "X": BLD / "python" / "Lesson_E" / "data" / "X.npy",
        "y": BLD / "python" / "Lesson_E" / "data" / "y.npy",
    },
    produces={
        "forest_model": BLD / "python" / "Lesson_E" / "model" / "forest_model.pkl",
        "R2_file": BLD / "python" / "Lesson_E" / "model_fit" / "R2_Forest.txt"
    }
):
    """Random forest."""
    X = np.load(depends_on["X"])
    y = np.load(depends_on["y"])
    forest_model, r2 = random_forest(X, y)
    with open(produces["forest_model"], "wb") as f:
        pickle.dump(forest_model, f)
    with open(produces["R2_file"], "w") as txt_file:
        txt_file.write(f"R2_Train: {r2}\n")
        

def task_load_and_split_data(
    produces={
        "X_train": BLD / "python" / "Lesson_E" / "data" / "X_train.npy",
        "X_test": BLD / "python" / "Lesson_E" / "data" / "X_test.npy",
        "y_train": BLD / "python" / "Lesson_E" / "data" / "y_train.npy",
        "y_test": BLD / "python" / "Lesson_E" / "data" / "y_test.npy",
    }
):
    """Load and split data."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    np.save(produces["X_train"], X_train)
    np.save(produces["X_test"], X_test)
    np.save(produces["y_train"], y_train)
    np.save(produces["y_test"], y_test)
    
    
def task_standardize_data(
    depends_on={
        "X_train": BLD / "python" / "Lesson_E" / "data" / "X_train.npy",
        "X_test": BLD / "python" / "Lesson_E" / "data" / "X_test.npy",
    },
    produces={
        "X_train_scaled": BLD / "python" / "Lesson_E" / "data" / "X_train_scaled.npy",
        "X_test_scaled": BLD / "python" / "Lesson_E" / "data" / "X_test_scaled.npy",
    }
):
    """Standardize data."""
    X_train = np.load(depends_on["X_train"])
    X_test = np.load(depends_on["X_test"])
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    np.save(produces["X_train_scaled"], X_train_scaled)
    np.save(produces["X_test_scaled"], X_test_scaled
)
    
    
def task_gradient_boosting_and_accuracy(
    depends_on={
        "X_train_scaled": BLD / "python" / "Lesson_E" / "data" / "X_train_scaled.npy",
        "y_train": BLD / "python" / "Lesson_E" / "data" / "y_train.npy",
        "X_test_scaled": BLD / "python" / "Lesson_E" / "data" / "X_test_scaled.npy",
        "y_test": BLD / "python" / "Lesson_E" / "data" / "y_test.npy",
    },
    produces={
        "gb_model": BLD / "python" / "Lesson_E" / "model" / "gb_model.pkl",
        "mse_file": BLD / "python" / "Lesson_E" / "model_fit" / "mse_GB.txt",
        "r2_file": BLD / "python" / "Lesson_E" / "model_fit" / "r2_GB.txt",
    }
):
    """Gradient boosting and accuracy."""
    X_train_scaled = np.load(depends_on["X_train_scaled"])
    y_train = np.load(depends_on["y_train"])
    X_test_scaled = np.load(depends_on["X_test_scaled"])
    y_test = np.load(depends_on["y_test"])
    gb_model, mse, r2 = gradient_boosting_and_accuracy(X_train_scaled, y_train, X_test_scaled, y_test)
    with open(produces["gb_model"], "wb") as f:
        pickle.dump(gb_model, f)
    with open(produces["mse_file"], "w") as txt_file:
        txt_file.write(f"mse_Train: {mse}\n")
    with open(produces["r2_file"], "w") as txt_file:
        txt_file.write(f"r2_Train: {r2}\n")



def task_determine_feature_importance(
    depends_on={
        "gb_model": BLD / "python" / "Lesson_E" / "model" / "gb_model.pkl",
        "x_test_std": BLD / "python" / "Lesson_E" / "data" / "X_test_scaled.npy",
        "y_test": BLD / "python" / "Lesson_E" / "data" / "y_test.npy",
    },
    produces={
        "feature_importance": BLD / "python" / "Lesson_E" / "figures" / "feature_importance.png",
    }
):
    """Determine feature importance."""
    gb_model = pickle.load(open(depends_on["gb_model"], "rb"))
    x_test_std = np.load(depends_on["x_test_std"])
    y_test = np.load(depends_on["y_test"])
    plt = determine_feature_importance(gb_model, x_test_std, y_test)
    plt.savefig(produces["feature_importance"])