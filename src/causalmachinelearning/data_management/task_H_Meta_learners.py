#Task file for H_Meta_learners

from pathlib import Path
from pytask import task
import pandas as pd
import numpy as np
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.H_Meta_learners import s_learner_fit, s_learner_predict, s_learner_evaluate_model, t_learner_fit, t_learner_cate, x_learner_fit, propensity_score_model, x_learner_st2, ps_predict, apply_ps_predict



def task_s_learner_fit(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
    },
    produces={
        "s_learner": BLD / "python" / "Lesson_H" / "model" / "s_learner.pkl",
    }
    ):
    """Train the s_learner model and save it as a pickle file"""
    df = pd.read_csv(depends_on["train"])
    s_learner = s_learner_fit(df, 3, 30)
    with open(produces["s_learner"], "wb") as f:
        pickle.dump(s_learner, f)
    
    
def task_s_learner_predict(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
        "test": SRC / "data" / "invest_email_rnd.csv",
        "s_learner": BLD / "python" / "Lesson_H" / "model" / "s_learner.pkl",
    },
    produces={
        "s_learner_cate_train": BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_train.npy",
        "s_learner_cate_test": BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_test.csv",
    }
    ):
    """Predict the cate for the train and test set"""
    train = pd.read_csv(depends_on["train"])
    test = pd.read_csv(depends_on["test"])
    with open(depends_on["s_learner"], "rb") as f:
        s_learner = pickle.load(f)
    s_learner_cate_train, s_learner_cate_test = s_learner_predict(train, test, s_learner)
    np.save(produces["s_learner_cate_train"], s_learner_cate_train)
    s_learner_cate_test.to_csv(produces["s_learner_cate_test"], index=False)


def task_evaluate_model(
    depends_on={
        "test": SRC / "data" / "invest_email_rnd.csv",
        "s_learner_cate_test": BLD / "python" / "Lesson_H" / "data" / "s_learner_cate_test.csv",
    },
    produces={
        "mse": BLD / "python" / "Lesson_H" / "model_fit" / "mse.txt",
        "mae": BLD / "python" / "Lesson_H" / "model_fit" / "mae.txt",
    }
    ):
    """Evaluate the model"""
    test = pd.read_csv(depends_on["test"])
    s_learner_cate_test = pd.read_csv(depends_on["s_learner_cate_test"])
    mse, mae = s_learner_evaluate_model(test, s_learner_cate_test)
    with open(produces["mse"], "w") as txt_file:
        txt_file.write(f"MSE: {mse}\n")
    with open(produces["mae"], "w") as txt_file:
        txt_file.write(f"MAE: {mae}\n")
        

def task_t_learner_fit(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
    },
    produces={
        "t_m0": BLD / "python" / "Lesson_H" / "model" / "t_m0.pkl",
        "t_m1": BLD / "python" / "Lesson_H" / "model" / "t_m1.pkl",
    }
    ):
    """Train the t_learner model and save it as a pickle file"""
    df = pd.read_csv(depends_on["train"])
    t_m0, t_m1 = t_learner_fit(df, 2, 60)
    with open(produces["t_m0"], "wb") as f:
        pickle.dump(t_m0, f)
    with open(produces["t_m1"], "wb") as f:
        pickle.dump(t_m1, f)
        
        
def task_t_learner_cate(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
        "test": SRC / "data" / "invest_email_rnd.csv",
        "t_m0": BLD / "python" / "Lesson_H" / "model" / "t_m0.pkl",
        "t_m1": BLD / "python" / "Lesson_H" / "model" / "t_m1.pkl",
    },
    produces={
        "t_learner_cate_train": BLD / "python" / "Lesson_H" / "data" / "t_learner_cate_train.npy",
        "t_learner_cate_test": BLD / "python" / "Lesson_H" / "data" / "t_learner_cate_test.csv",
    }
    ):
    """Predict the cate for the train and test set"""
    train = pd.read_csv(depends_on["train"])
    test = pd.read_csv(depends_on["test"])
    with open(depends_on["t_m0"], "rb") as f:
        t_m0 = pickle.load(f)
    with open(depends_on["t_m1"], "rb") as f:
        t_m1 = pickle.load(f)
    t_learner_cate_train, t_learner_cate_test = t_learner_cate(train, test, t_m0, t_m1)
    np.save(produces["t_learner_cate_train"], t_learner_cate_train)
    t_learner_cate_test.to_csv(produces["t_learner_cate_test"], index=False)
    

def task_x_learner_fit(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
    },
    produces={
        "x1_m0": BLD / "python" / "Lesson_H" / "model" / "x1_m0.pkl",
        "x1_m1": BLD / "python" / "Lesson_H" / "model" / "x1_m1.pkl",
    }
    ):
    """Train the x_learner model and save it as a pickle file"""
    df = pd.read_csv(depends_on["train"])
    x1_m0, x1_m1 = x_learner_fit(df, 2, 30)
    with open(produces["x1_m0"], "wb") as f:
        pickle.dump(x1_m0, f)
    with open(produces["x1_m1"], "wb") as f:
        pickle.dump(x1_m1, f)
        
        
def task_propensity_score_model(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
    },
    produces={
        "g": BLD / "python" / "Lesson_H" / "model" / "g.pkl",
    }
    ):
    """Train the propensity score model and save it as a pickle file"""
    df = pd.read_csv(depends_on["train"])
    g = propensity_score_model(df)
    with open(produces["g"], "wb") as f:
        pickle.dump(g, f)
        
        
def task_x_learner_st2(
    depends_on={
        "train": SRC / "data" / "invest_email_biased.csv",
        "x1_m0": BLD / "python" / "Lesson_H" / "model" / "x1_m0.pkl",
        "x1_m1": BLD / "python" / "Lesson_H" / "model" / "x1_m1.pkl",
    },
    produces={
        "x2_m0": BLD / "python" / "Lesson_H" / "model" / "x2_m0.pkl",
        "x2_m1": BLD / "python" / "Lesson_H" / "model" / "x2_m1.pkl",
        "d_train": BLD / "python" / "Lesson_H" / "data" / "d_train.npy",
    }
    ):
    """Train the second stage models on the imputed treatment effects, split by treatment variable"""
    train = pd.read_csv(depends_on["train"])
    with open(depends_on["x1_m0"], "rb") as f:
        x1_m0 = pickle.load(f)
    with open(depends_on["x1_m1"], "rb") as f:
        x1_m1 = pickle.load(f)
    x2_m0, x2_m1, d_train = x_learner_st2(train, x1_m0, x1_m1)
    with open(produces["x2_m0"], "wb") as f:
        pickle.dump(x2_m0, f)
    with open(produces["x2_m1"], "wb") as f:
        pickle.dump(x2_m1, f)
    np.save(produces["d_train"], d_train)
    
    
def task_apply_ps_predict(
    depends_on={
        "test": SRC / "data" / "invest_email_rnd.csv",
        "train": SRC / "data" / "invest_email_biased.csv",
        "g": BLD / "python" / "Lesson_H" / "model" / "g.pkl",
        "x2_m0": BLD / "python" / "Lesson_H" / "model" / "x2_m0.pkl",
        "x2_m1": BLD / "python" / "Lesson_H" / "model" / "x2_m1.pkl",
    },
    produces={
        "x_learner_cate_test": BLD / "python" / "Lesson_H" / "data" / "x_learner_cate_test.csv",
        "x_learner_cate_train": BLD / "python" / "Lesson_H" / "data" / "x_learner_cate_train.npy",
    }
    ):
    """Use the propensity score model to predict the cate"""
    test = pd.read_csv(depends_on["test"])
    train = pd.read_csv(depends_on["train"])
    with open(depends_on["g"], "rb") as f:
        g = pickle.load(f)
    with open(depends_on["x2_m0"], "rb") as f:
        x2_m0 = pickle.load(f)
    with open(depends_on["x2_m1"], "rb") as f:
        x2_m1 = pickle.load(f)
    x_learner_cate_train, x_learner_cate_test = apply_ps_predict(train, test, g, x2_m0, x2_m1)
    x_learner_cate_test.to_csv(produces["x_learner_cate_test"], index=False)
    np.save(produces["x_learner_cate_train"], x_learner_cate_train)