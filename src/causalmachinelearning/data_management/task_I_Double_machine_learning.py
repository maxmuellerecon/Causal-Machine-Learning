#Task file for I_Double_machine_learning

from pathlib import Path
from pytask import task
import pandas as pd
import numpy as np
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.I_Double_machine_learning import plot_pattern, fwl_theorem, verify_fwl_theorem, debias_treatment, plot_debiased, denoise_outcome


def task_plot_pattern(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
    },
    produces={
        "plot": BLD / "python" / "Lesson_I" / "figures" / "plot.png",
    }
    ):
    """Plot pattern"""
    train = pd.read_csv(depends_on["train"])
    plot = plot_pattern(train, "price", "sales")
    plot.savefig(produces["plot"])
    

def task_fwl_theorem(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
    },
    produces={
        "table1": BLD / "python" / "Lesson_I" / "tables" / "table1.tex",
    }
    ): 
    """Apply the Frisch-Waugh-Lovell Theorem"""
    train = pd.read_csv(depends_on["train"])
    table1 = fwl_theorem(train)
    # Export to LaTeX
    latex_code = table1.as_latex()
    print(latex_code)
    with open(produces["table1"], 'w') as f:
        f.write(latex_code)
        
        
def task_verify_fwl_theorem(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
    },
    produces={
        "table2": BLD / "python" / "Lesson_I" / "tables" / "table2.tex",
    }
    ): 
    """Verify that FWL is the same as usual regression"""
    train = pd.read_csv(depends_on["train"])
    table2 = verify_fwl_theorem(train)
    # Export to LaTeX
    latex_code = table2.as_latex()
    print(latex_code)
    with open(produces["table2"], 'w') as f:
        f.write(latex_code)
        
        
def task_debias_treatment(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
    },
    produces={
        "train_pred": BLD / "python" / "Lesson_I" / "data" / "train_pred.csv",
    }
    ):
    """Debias the treatment"""
    train = pd.read_csv(depends_on["train"])
    train_pred = debias_treatment(train)
    train_pred.to_csv(produces["train_pred"], index=False)
    

def task_plot_debiased(
    depends_on={
        "train_pred": BLD / "python" / "Lesson_I" / "data" / "train_pred.csv",
    },
    produces={
        "plot": BLD / "python" / "Lesson_I" / "figures" / "plot_debiased.png",
    }
    ):
    """Plot debiased scatterplot"""
    train_pred = pd.read_csv(depends_on["train_pred"])
    plot = plot_debiased(train_pred)
    plot.savefig(produces["plot"])
    

def task_denoise_outcome(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
        "train_pred": BLD / "python" / "Lesson_I" / "data" / "train_pred.csv",
    },
    produces={
        "train_pred_y": BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv",
    }
    ):
    """Denoise the outcome"""
    train = pd.read_csv(depends_on["train"])
    train_pred = pd.read_csv(depends_on["train_pred"])
    train_pred_y = denoise_outcome(train, train_pred)
    train_pred_y.to_csv(produces["train_pred_y"], index=False)