# J_Diff_and_diff_and_ml

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

style.use("ggplot")

from causalmachinelearning.lessons.__exceptions import _fail_if_not_dataframe


################################J.1 Two-way Fixed Effects################################
# Create the data
def create_data():
    """Create data."""
    date = pd.date_range("2021-05-01", "2021-07-31", freq="D")
    cohorts = pd.to_datetime(["2021-06-01", "2021-07-15", "2022-01-01"]).date
    units = range(1, 100 + 1)
    np.random.seed(1)
    data = (
        pd.DataFrame(
            dict(
                date=np.tile(date, len(units)),
                unit=np.repeat(units, len(date)),
                cohort=np.repeat(np.random.choice(cohorts, len(units)), len(date)),
                unit_fe=np.repeat(np.random.normal(0, 5, size=len(units)), len(date)),
                time_fe=np.tile(np.random.normal(size=len(date)), len(units)),
                week_day=np.tile(date.weekday, len(units)),
                w_seas=np.tile(abs(5 - date.weekday) % 7, len(units)),
            )
        )
        .assign(
            trend=lambda d: (d["date"] - d["date"].min()).dt.days / 70,
            day=lambda d: (d["date"] - d["date"].min()).dt.days,
            treat=lambda d: (d["date"] >= d["cohort"]).astype(int),
        )
        .assign(
            y0=lambda d: 10
            + d["trend"]
            + d["unit_fe"]
            + 0.1 * d["time_fe"]
            + d["w_seas"] / 10,
        )
        .assign(y1=lambda d: d["y0"] + 1)
        .assign(
            tau=lambda d: d["y1"] - d["y0"],
            installs=lambda d: np.where(d["treat"] == 1, d["y1"], d["y0"]),
        )
    )
    return data


def plot_trend(data):
    """Plot trend of outcome variable."""
    _fail_if_not_dataframe(data)

    cohort_dates = pd.to_datetime(["2021-06-01", "2021-07-15", "2022-01-01"]).date
    plt.figure(figsize=(10, 4))
    [
        plt.vlines(x=cohort, ymin=9, ymax=15, color=color, ls="dashed")
        for color, cohort in zip(["C0", "C1"], cohort_dates[:-1])
    ]
    sns.lineplot(
        data=(data.groupby(["cohort", "date"])["installs"].mean().reset_index()),
        x="date",
        y="installs",
        hue="cohort",
    )
    return plt


def twfe_regression(df):
    """Run two-way fixed effects regression.

    Args:
        df (pd.DataFrame): data

    Returns:
        twfe_output (str): summary of regression
        twfe_model (statsmodels.regression.linear_model.RegressionResultsWrapper): regression model

    """
    _fail_if_not_dataframe(df)

    formula = f"""installs ~ treat + C(unit) + C(date)"""
    twfe_output = smf.ols(formula, data=df).fit().summary()
    twfe_model = smf.ols(formula, data=df).fit()
    # Since data is simulated, TWFE should recover the treatment effect on the treated, which is exactly one
    df.query("treat == 1")["tau"].mean()
    return twfe_output, twfe_model


def plot_counterfactuals(df, twfe_model):
    """Plot counterfactuals (what happens, if they had not been treated)"""
    _fail_if_not_dataframe(df)

    df_pred = df.assign(
        **{"installs_hat_0": twfe_model.predict(df.assign(**{"treat": 0}))}
    )
    plt.figure(figsize=(10, 4))
    cohorts = pd.to_datetime(["2021-06-01", "2021-07-15", "2022-01-01"]).date
    [
        plt.vlines(x=cohort, ymin=9, ymax=15, color=color, ls="dashed")
        for color, cohort in zip(["C0", "C1"], cohorts[:-1])
    ]
    sns.lineplot(
        data=(
            df_pred.groupby(["cohort", "date"])["installs_hat_0"].mean().reset_index()
        ),
        x="date",
        y="installs_hat_0",
        hue="cohort",
        alpha=0.7,
        ls="dotted",
        legend=None,
    )
    sns.lineplot(
        data=(df_pred.groupby(["cohort", "date"])["installs"].mean().reset_index()),
        x="date",
        y="installs",
        hue="cohort",
    )
    return plt


################################J.2 Time Treatment Heterogeneity ################################
# Recently, we got to know that TWFE are severely biased.
# Especially heterogeneous treatment effects over time
def g_plot_data_fct(df):
    """Create data for plot."""
    _fail_if_not_dataframe(df)

    g_plot_data = (
        df.groupby(["cohort", "date"])["installs"]
        .mean()
        .reset_index()
        .astype({"cohort": str})
    )
    return g_plot_data


def plot_comparison(data):
    """Plot comparison of cohorts."""
    _fail_if_not_dataframe(data)

    cohorts = data["cohort"].unique()
    palette = dict(zip(map(str, cohorts), ["C0", "C1", "C2"]))
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)

    def plot_comp(df, ax, exclude_cohort, name):
        sns.lineplot(
            data=df.query(f"cohort != '{exclude_cohort}'"),
            x="date",
            y="installs",
            hue="cohort",
            palette=palette,
            legend=None,
            ax=ax,
        )
        sns.lineplot(
            data=df.query(f"cohort == '{exclude_cohort}'"),
            x="date",
            y="installs",
            hue="cohort",
            palette=palette,
            alpha=0.2,
            legend=None,
            ax=ax,
        )
        ax.set_title(name)

    plot_comp(data, axs[0, 0], cohorts[1], "Early vs Never")
    plot_comp(data, axs[0, 1], cohorts[0], "Late vs Never")
    plot_comp(data[data["date"] <= cohorts[1]], axs[1, 0], cohorts[-1], "Early vs Late")
    plot_comp(data[data["date"] > cohorts[0]], axs[1, 1], cohorts[-1], "Late vs Early")
    plt.tight_layout()
    return plt


def fct_late_vs_early(df_heter):
    """Use data from 2021-06-01 to 2021-08-01 to compare late vs early."""
    _fail_if_not_dataframe(df_heter)

    late_vs_early = df_heter[df_heter["date"].astype(str) >= "2021-06-01"][
        lambda d: d["cohort"].astype(str) <= "2021-08-01"
    ]
    return late_vs_early


# However, we will always have a bias, since Goodman-Bacon showed that we have downward bias if the magnitude of the effect increases with time
# Will see: effect of TWFE is smaller than true ATT
# Since we have different timing in the treatment, the early treated gets used as a control for the late treated units, which distorts counterfactuals
# Also including leads and lags does not solve the problem
def plot_twfe_regression_late_vs_early(late_vs_early):
    """Plot TWFE regression for late vs early with counterfactual for late group."""
    _fail_if_not_dataframe(late_vs_early)

    formula = f"""installs ~ treat + C(date) + C(unit)"""
    twfe_model = smf.ols(formula, data=late_vs_early).fit()
    late_vs_early_pred = (
        late_vs_early.assign(
            **{
                "installs_hat_0": twfe_model.predict(
                    late_vs_early.assign(**{"treat": 0})
                )
            }
        )
        .groupby(["cohort", "date"])[["installs", "installs_hat_0"]]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(10, 4))
    plt.title("Late vs Early Counterfactuals")
    sns.lineplot(
        data=late_vs_early_pred, x="date", y="installs", hue="cohort", legend=None
    )
    sns.lineplot(
        data=(
            late_vs_early_pred[
                late_vs_early_pred["cohort"].astype(str) == "2021-07-15"
            ][lambda d: d["date"].astype(str) >= "2021-07-15"]
        ),
        x="date",
        y="installs_hat_0",
        alpha=0.7,
        color="C0",
        ls="dotted",
        label="counterfactual",
    )
    return plt


################################J.3 Flexible Functional Forms################################
def twfe_regression_groups(df_heter):
    """Heterogeneous treatment effects by cohort/group."""
    _fail_if_not_dataframe(df_heter)

    formula = f"""installs ~ treat:C(cohort):C(date) + C(unit) + C(date)"""
    # for nicer plots latter on
    df_heter_str = df_heter.astype({"cohort": str, "date": str})
    twfe_model_groups = smf.ols(formula, data=df_heter_str).fit()
    return twfe_model_groups, df_heter_str


def check_trueATT_vs_predATT(df_heter_str, twfe_model_groups):
    """Check true ATT vs predicted ATT in this case.
    Args:
        df_heter_str (pd.DataFrame): data
        twfe_model_groups (statsmodels.regression.linear_model.RegressionResultsWrapper): regression model

    Returns:
        df_pred (pd.DataFrame): data with predicted ATT
        length (int): length of regression output
        tau_mean (float): mean of true ATT
        pred_effect_mean (float): mean of predicted ATT
    """
    _fail_if_not_dataframe(df_heter_str)

    df_pred = df_heter_str.assign(
        **{
            "installs_hat_0": twfe_model_groups.predict(
                df_heter_str.assign(**{"treat": 0})
            )
        }
    ).assign(**{"effect_hat": lambda d: d["installs"] - d["installs_hat_0"]})
    length = len(twfe_model_groups.params)
    tau_mean = df_pred.query("treat==1")["tau"].mean()
    pred_effect_mean = df_pred.query("treat==1")["effect_hat"].mean()
    return df_pred, length, tau_mean, pred_effect_mean


def plot_treatment_effect(twfe_model_groups):
    """Plot treatment effect by cohort."""
    effects = (
        twfe_model_groups.params[twfe_model_groups.params.index.str.contains("treat")]
        .reset_index()
        .rename(columns={0: "param"})
        .assign(cohort=lambda d: d["index"].str.extract(r"C\(cohort\)\[(.*)\]:"))
        .assign(date=lambda d: d["index"].str.extract(r":C\(date\)\[(.*)\]"))
        .assign(
            date=lambda d: pd.to_datetime(d["date"]),
            cohort=lambda d: pd.to_datetime(d["cohort"]),
        )
    )
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=effects, x="date", y="param", hue="cohort")
    plt.xticks(rotation=45)
    plt.ylabel("Estimated Effect")
    return plt
