#Task file for G_Treatment_effect_estimators

import pandas as pd
import numpy as np
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.G_Treatment_effect_estimators import split_train_test, create_y_star, train_model, predict_cate, split_train_test_ice, predict_y_star_cont


def task_split_train_test(
    depends_on={
        "data": SRC / "data" / "invest_email_rnd.csv",
    },
    produces={
        "train": BLD / "python" / "Lesson_G" / "data" / "train.csv",
        "test": BLD / "python" / "Lesson_G" / "data" / "test.csv",
    }
    ):
    """Split data into train and test sets"""
    data = pd.read_csv(depends_on["data"])
    train, test = split_train_test(data, 0.4)
    train.to_csv(produces["train"], index=False)
    test.to_csv(produces["test"], index=False)
    
    
def task_create_y_star(
    depends_on={
        "train": BLD / "python" / "Lesson_G" / "data" / "train.csv",
    },
    produces={
        "y_star_train": BLD / "python" / "Lesson_G" / "data" / "y_star_train.csv",
        "ps": BLD / "python" / "Lesson_G" / "data" / "ps.npy",
    }
):
    """Create y_star"""
    train = pd.read_csv(depends_on["train"])
    y_star_train, ps = create_y_star(train)
    y_star_train.to_csv(produces["y_star_train"], index=False)
    np.save(produces["ps"], ps)
    
    
def task_train_model(
    depends_on={
        "train": BLD / "python" / "Lesson_G" / "data" / "train.csv",
        "y_star_train": BLD / "python" / "Lesson_G" / "data" / "y_star_train.csv",
    },
    produces={
        "cate_learner": BLD / "python" / "Lesson_G" / "model" / "cate_learner.pkl",
    }
):
    """Train model"""
    train = pd.read_csv(depends_on["train"])
    y_star_train = pd.read_csv(depends_on["y_star_train"])
    cate_learner = train_model(train, y_star_train)
    with open(produces["cate_learner"], "wb") as f:
        pickle.dump(cate_learner, f)
       
        
def task_predict_cate(
    depends_on={
        "test": BLD / "python" / "Lesson_G" / "data" / "test.csv",
        "cate_learner": BLD / "python" / "Lesson_G" / "model" / "cate_learner.pkl",
    },
    produces={
        "test_pred": BLD / "python" / "Lesson_G" / "data" / "test_pred.csv",
    }
):
    """Predict cate"""
    test = pd.read_csv(depends_on["test"])
    with open(depends_on["cate_learner"], "rb") as f:
        cate_learner = pickle.load(f)
    test_pred = predict_cate(test, cate_learner)
    test_pred.to_csv(produces["test_pred"], index=False)
    

def task_split_train_test_ice(
    depends_on={
        "data": SRC / "data" / "ice_cream_sales_rnd.csv",
    },
    produces={
        "train": BLD / "python" / "Lesson_G" / "data" / "train_ice.csv",
        "test": BLD / "python" / "Lesson_G" / "data" / "test_ice.csv",
    }
    ):
    """Split data into train and test sets"""
    data = pd.read_csv(depends_on["data"])
    train, test = split_train_test_ice(data, 0.3)
    train.to_csv(produces["train"], index=False)
    test.to_csv(produces["test"], index=False)
    
    
def task_predict_y_star_cont(
    depends_on={
        "train": BLD / "python" / "Lesson_G" / "data" / "train_ice.csv",
        "test": BLD / "python" / "Lesson_G" / "data" / "test_ice.csv",
    },
    produces={
        "test_pred": BLD / "python" / "Lesson_G" / "data" / "test_pred_ice.csv",
    }
    ):
    """Predict y_star for continuous treatment"""
    train = pd.read_csv(depends_on["train"])
    test = pd.read_csv(depends_on["test"])
    test_pred = predict_y_star_cont(train, test)
    test_pred.to_csv(produces["test_pred"], index=False)