#Task File for F_Heterogeneous Treatment Effects

from pathlib import Path
from pytask import task
import pandas as pd
import numpy as np
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.F_Heterogeneous_treatment_effects import split_data, regressions_three, pred_sensitivity, ml_model, comparison, plot_regr_model, plot_ml_model


def task_split_data(   
        depends_on={
            "data": SRC / "data" / "ice_cream_sales_rnd.csv",
        },
        produces={
            "train": BLD / "python" / "Lesson_F" / "data" / "train.csv",
            "test": BLD / "python" / "Lesson_F" / "data" / "test.csv",
        }
        ):
    """Split data into train and test."""
    data = pd.read_csv(depends_on["data"])
    train, test = split_data(data)
    train.to_csv(produces["train"])
    test.to_csv(produces["test"])
    

def task_regressions_three(
    depends_on={
        "train": BLD / "python" / "Lesson_F" / "data" / "train.csv",
    },
    produces={
        "m1": BLD / "python" / "Lesson_F" / "model" / "m1.pkl",
        "m2": BLD / "python" / "Lesson_F" / "model" / "m2.pkl",
        "m3": BLD / "python" / "Lesson_F" / "model" / "m3.pkl",
        "latex_code1": BLD / "python" / "Lesson_F" / "tables" / "latex_code1.txt",
        "latex_code2": BLD / "python" / "Lesson_F" / "tables" / "latex_code2.txt",
        "latex_code3": BLD / "python" / "Lesson_F" / "tables" / "latex_code3.txt",
    }
):
    """Run linear regression models."""
    train = pd.read_csv(depends_on["train"])
    m1, m2, m3, latex_code1, latex_code2, latex_code3 = regressions_three(train)
    with open(produces["m1"], "wb") as f:
        pickle.dump(m1, f)
    with open(produces["m2"], "wb") as f:
        pickle.dump(m2, f)
    with open(produces["m3"], "wb") as f:
        pickle.dump(m3, f)
    with open(produces["latex_code1"], "w") as txt_file:
        txt_file.write(latex_code1)
    with open(produces["latex_code2"], "w") as txt_file:
        txt_file.write(latex_code2)
    with open(produces["latex_code3"], "w") as txt_file:
        txt_file.write(latex_code3)


def task_pred_sensitivity(
    depends_on={
        "test": BLD / "python" / "Lesson_F" / "data" / "test.csv",
        "m1": BLD / "python" / "Lesson_F" / "model" / "m1.pkl",
        "m3": BLD / "python" / "Lesson_F" / "model" / "m3.pkl",
    },
    produces={
        "pred_sens_m1": BLD / "python" / "Lesson_F" / "data" / "pred_sens_m1.csv",
        "pred_sens_m3": BLD / "python" / "Lesson_F" / "data" / "pred_sens_m3.csv",
    }
):
    """Predict sensitivity."""
    test = pd.read_csv(depends_on["test"])
    with open(depends_on["m1"], "rb") as f:
        m1 = pickle.load(f)
    with open(depends_on["m3"], "rb") as f:
        m3 = pickle.load(f)
    pred_sens_m1 = pred_sensitivity(m1, test)
    pred_sens_m3 = pred_sensitivity(m3, test)   
    pred_sens_m1.to_csv(produces["pred_sens_m1"])
    pred_sens_m3.to_csv(produces["pred_sens_m3"])
    

def task_ml_model(
    depends_on={
        "train": BLD / "python" / "Lesson_F" / "data" / "train.csv",
        "test": BLD / "python" / "Lesson_F" / "data" / "test.csv",
    },
    produces={
        "m4": BLD / "python" / "Lesson_F" / "model" / "m4.pkl",
    }
):
    """Run machine learning model."""
    train = pd.read_csv(depends_on["train"])
    test = pd.read_csv(depends_on["test"])
    model = ml_model(train, test)
    with open(produces["m4"], "wb") as f:
        pickle.dump(model, f)
        

def task_comparison(
    depends_on={
        "regr_data": BLD / "python" / "Lesson_F" / "data" / "pred_sens_m3.csv",
        "m4": BLD / "python" / "Lesson_F" / "model" / "m4.pkl",
    },
    produces={
        "bands_df": BLD / "python" / "Lesson_F" / "data" / "bands_df.csv",
    },
):  
    """Compare regression and machine learning model."""
    regr_data = pd.read_csv(depends_on["regr_data"])
    with open(depends_on["m4"], "rb") as f:
        m4 = pickle.load(f)
    bands_df = comparison(regr_data, m4, 2)
    bands_df.to_csv(produces["bands_df"])
    

def task_plot_regr_model(
    depends_on={
        "bands_df": BLD / "python" / "Lesson_F" / "data" / "bands_df.csv",
    },
    produces={
        "plot_reg": BLD / "python"  / "Lesson_F"/ "figures"  / "plot_reg.png",
    },
):  
    """Plot regression model."""
    bands_df = pd.read_csv(depends_on["bands_df"])
    plotreg = plot_regr_model(bands_df)
    plotreg.savefig(produces["plot_reg"])


def task_plot_ml_model(
    depends_on={
        "bands_df": BLD / "python" / "Lesson_F" / "data" / "bands_df.csv",
    },
    produces={
        "plot_ml": BLD / "python"  / "Lesson_F" / "figures"  / "plot_ml.png",
    },
):  
    """Plot machine learning model."""
    bands_df = pd.read_csv(depends_on["bands_df"])
    plotml = plot_ml_model(bands_df)
    plotml.savefig(produces["plot_ml"])
    
