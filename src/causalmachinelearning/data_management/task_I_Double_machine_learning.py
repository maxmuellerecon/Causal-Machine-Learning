#Task file for I_Double_machine_learning

from pathlib import Path
from pytask import task
import pandas as pd
import numpy as np
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.data_management.I_Double_machine_learning import plot_pattern, fwl_theorem, verify_fwl_theorem, debias_treatment, plot_debiased, denoise_outcome, plot_debiased_denoised, comparison_models, parametric_double_ml_cate, orthogonalize_treatment_and_outcome, non_parametric_double_ml_cate


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
    

def task_plot_debiased_denoised(
    depends_on={
        "train_pred_y": BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv",
    },
    produces={
        "plot_debiased_denoised": BLD / "python" / "Lesson_I" / "figures" / "plot_debiased_denoised.png",
    }
    ):
    """Plot debiased and denoised scatterplot"""
    train_pred_y = pd.read_csv(depends_on["train_pred_y"])
    plot = plot_debiased_denoised(train_pred_y)
    plot.savefig(produces["plot_debiased_denoised"])
    

def task_comparison_models(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
        "train_pred_y": BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv",
    },
    produces={
        "final_model": BLD / "python" / "Lesson_I" / "tables" / "final_model.tex",
        "basic_model": BLD / "python" / "Lesson_I" / "tables" / "basic_model.tex",
    }
    ):
    """Compare the models"""
    train = pd.read_csv(depends_on["train"])
    train_pred_y = pd.read_csv(depends_on["train_pred_y"])
    final_model, basic_model = comparison_models(train_pred_y, train)
    # Export to LaTeX
    latex_code = final_model.as_latex()
    print(latex_code)
    with open(produces["final_model"], 'w') as f:
        f.write(latex_code)
    # Append to LaTeX for basic model
    latex_code_basic = basic_model.as_latex()
    print(latex_code_basic)
    with open(produces["basic_model"], 'a') as f:
        f.write(latex_code_basic)
        

def task_parametric_double_ml_cate(
    depends_on={
        "test": SRC / "data" / "ice_cream_sales_rnd.csv",
        "train_pred_y": BLD / "python" / "Lesson_I" / "data" / "train_pred_y.csv",
    },
    produces={
        "final_model_cate": BLD / "python" / "Lesson_I" / "models" / "final_model_cate.pkl",
        "cate_test": BLD / "python" / "Lesson_I" / "data" / "cate_test.csv",
    }
    ):
    """Parametric Double ML CATE"""
    test = pd.read_csv(depends_on["test"])
    train_pred_y = pd.read_csv(depends_on["train_pred_y"])
    final_model_cate, cate_test = parametric_double_ml_cate(train_pred_y, test)
    #Save the model
    with open(produces["final_model_cate"], "wb") as f:
        pickle.dump(final_model_cate, f)
    #Save the CATE
    cate_test.to_csv(produces["cate_test"], index=False)
    
    
def task_orthogonalize_treatment_and_outcome(
    depends_on={
        "train": SRC / "data" / "ice_cream_sales.csv",
    },
    produces={
        "train_pred_nonparam": BLD / "python" / "Lesson_I" / "data" / "train_pred_nonparam.csv",
    }
    ):
    """Orthogonalize treatment and outcome"""
    train = pd.read_csv(depends_on["train"])
    train_pred_nonparam = orthogonalize_treatment_and_outcome(train)
    train_pred_nonparam.to_csv(produces["train_pred_nonparam"], index=False)
    

def task_non_parametric_double_ml_cate(
    depends_on={
        "test": SRC / "data" / "ice_cream_sales_rnd.csv",
        "train_pred_nonparam": BLD / "python" / "Lesson_I" / "data" / "train_pred_nonparam.csv",
    },
    produces={
        "cate_test_nonparam": BLD / "python" / "Lesson_I" / "data" / "cate_test_nonparam.csv",
    }
    ):
    """Non-parametric Double ML CATE"""
    test = pd.read_csv(depends_on["test"])
    train_pred_nonparam = pd.read_csv(depends_on["train_pred_nonparam"])
    cate_test_nonparam = non_parametric_double_ml_cate(train_pred_nonparam, test)
    cate_test_nonparam.to_csv(produces["cate_test_nonparam"], index=False)