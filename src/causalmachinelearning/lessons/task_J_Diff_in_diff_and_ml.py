#Task file for J_Diff_in_diff_and_ml

import pandas as pd
import pickle

from causalmachinelearning.config import BLD, SRC
from causalmachinelearning.lessons.J_Diff_and_diff_and_ml import create_data, plot_trend, twfe_regression, plot_counterfactuals, g_plot_data_fct, plot_comparison, fct_late_vs_early, plot_twfe_regression_late_vs_early, twfe_regression_groups, check_trueATT_vs_predATT, plot_treatment_effect


def task_create_data(
    produces={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
    }
    ):
    """Create data"""
    data = create_data()
    data.to_csv(produces["data"], index=False)


def task_plot_trend(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
    },
    produces={
        "plot_trend": BLD / "python" / "Lesson_J" / "figures" / "plot_trend.png",
    }
    ):
    """Plot trend of outcome variable"""
    data = pd.read_csv(depends_on["data"])
    plt = plot_trend(data)
    plt.savefig(produces["plot_trend"], dpi=300)
    

def task_twfe_regression(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
    },
    produces={
        "TWFE_output": BLD / "python" / "Lesson_J" / "tables" / "TWFE_output.tex",
        "TWFE_model": BLD / "python" / "Lesson_J" / "model" / "TWFE_model.pkl",
    }
    ):
    """Run two-way fixed effects regression"""
    data = pd.read_csv(depends_on["data"])
    twfe_output, twfe_model = twfe_regression(data)
    with open(produces["TWFE_model"], "wb") as f:
        pickle.dump(twfe_model, f)
    with open(produces["TWFE_output"], "w") as f:
        f.write(twfe_output.as_latex())
  
    
def task_plot_counterfactuals(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
        "TWFE_model": BLD / "python" / "Lesson_J" / "model" / "TWFE_model.pkl",
    },
    produces={
        "plot_counterfactuals": BLD / "python" / "Lesson_J" / "figures" / "plot_counterfactuals.png",
    }
    ):
    """Plot counterfactuals"""
    data = pd.read_csv(depends_on["data"])
    with open(depends_on["TWFE_model"], "rb") as f:
        twfe_model = pickle.load(f)
    plt = plot_counterfactuals(data, twfe_model)
    plt.savefig(produces["plot_counterfactuals"], dpi=300)
    

def task_g_plot_data_fct(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
    },
    produces={
        "g_plot_data": BLD / "python" / "Lesson_J" / "data" / "g_plot_data.csv",
    }
    ):
    """group data"""
    data = pd.read_csv(depends_on["data"])
    g_plot_data = g_plot_data_fct(data)
    g_plot_data.to_csv(produces["g_plot_data"], index=False)
    
    
def task_plot_comparison(
    depends_on={
        "g_plot_data": BLD / "python" / "Lesson_J" / "data" / "g_plot_data.csv",
    },
    produces={
        "plot_comparison": BLD / "python" / "Lesson_J" / "figures" / "plot_comparison.png",
    }
    ):
    """Plot comparison"""
    g_plot_data = pd.read_csv(depends_on["g_plot_data"])
    plt = plot_comparison(g_plot_data)
    plt.savefig(produces["plot_comparison"], dpi=300)
    
    
def task_fct_late_vs_early(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
    },
    produces={
        "data_late_vs_early": BLD / "python" / "Lesson_J" / "data" / "data_late_vs_early.csv",
    }
    ):
    """Create data for late vs early"""
    data = pd.read_csv(depends_on["data"])
    data_late_vs_early_data = fct_late_vs_early(data)
    data_late_vs_early_data.to_csv(produces["data_late_vs_early"], index=False)
    
    
def task_plot_twfe_regression_late_vs_early(
    depends_on={
        "data_late_vs_early": BLD / "python" / "Lesson_J" / "data" / "data_late_vs_early.csv",
    },
    produces={
        "plot_twfe_regression_late_vs_early": BLD / "python" / "Lesson_J" / "figures" / "plot_twfe_regression_late_vs_early.png",
    }
    ):
    """Plot twfe regression late vs early"""
    data_late_vs_early = pd.read_csv(depends_on["data_late_vs_early"])
    plt = plot_twfe_regression_late_vs_early(data_late_vs_early)
    plt.savefig(produces["plot_twfe_regression_late_vs_early"], dpi=300)
    

def task_twfe_regression_groups(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
    },
    produces={
        "TWFE_model_groups": BLD / "python" / "Lesson_J" / "model" / "TWFE_model_groups.pkl",
        "df_heter_str": BLD / "python" / "Lesson_J" / "data" / "df_heter_str.csv",
    }
    ):
    """"Run two-way fixed effects regression for groups"""
    data = pd.read_csv(depends_on["data"])
    twfe_model_groups, df_heter_str = twfe_regression_groups(data)
    df_heter_str.to_csv(produces["df_heter_str"], index=False)
    with open(produces["TWFE_model_groups"], "wb") as f:
        pickle.dump(twfe_model_groups, f)
        
        
def task_check_trueATT_vs_predATT(
    depends_on={
        "data": BLD / "python" / "Lesson_J" / "data" / "data.csv",
        "TWFE_model_groups": BLD / "python" / "Lesson_J" / "model" / "TWFE_model_groups.pkl",
    },
    produces={
        "df_pred": BLD / "python" / "Lesson_J" / "data" / "df_pred.csv",
        "output": BLD / "python" / "Lesson_J" / "model_fit" / "output.tex",
    }
    ):
    """Check true ATT vs pred ATT"""
    data = pd.read_csv(depends_on["data"])
    with open(depends_on["TWFE_model_groups"], "rb") as f:
        twfe_model_groups = pickle.load(f)
    df_pred, length, tau_mean, pred_effect_mean = check_trueATT_vs_predATT(data, twfe_model_groups)
    with open(produces["output"], "w") as tex_file:
        tex_file.write("\\documentclass{article}\n")
        tex_file.write("\\begin{document}\n\n")
        
        tex_file.write("\\section*{Results}\n")
        tex_file.write("\\begin{itemize}\n")
        
        tex_file.write("\\item Number of param.: " + str(length) + "\n")
        tex_file.write("\\item True Effect: " + str(tau_mean) + "\n")
        tex_file.write("\\item Pred. Effect: " + str(pred_effect_mean) + "\n")
        
        tex_file.write("\\end{itemize}\n\n")
        tex_file.write("\\end{document}\n")
    df_pred.to_csv(produces["df_pred"], index=False)
    
    
def task_plot_treatment_effect(
    depends_on={
        "TWFE_model_groups": BLD / "python" / "Lesson_J" / "model" / "TWFE_model_groups.pkl",
    },
    produces={
        "plot_treatment_effect": BLD / "python" / "Lesson_J" / "figures" / "plot_treatment_effect.png",
    }
    ):
    """Plot treatment effect"""
    with open(depends_on["TWFE_model_groups"], "rb") as f:
        twfe_model_groups = pickle.load(f)
    plt = plot_treatment_effect(twfe_model_groups)
    plt.savefig(produces["plot_treatment_effect"], dpi=300)