# causalmachinelearning

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/maxmuellerecon/causalmachinelearning/main.svg)](https://results.pre-commit.ci/latest/github/maxmuellerecon/causalmachinelearning/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To get started, create and activate the environment with
```console
$ conda/mamba env create
$ conda activate cml
```

To build the project, type
```console
$ pytask
```

## Description

This contains the code for the final project of the "EPP" course at the University of Bonn. \
However, this repository is also a starting point for people interested in causal machine learning in python.

## Structure

The class is structured as follows: \
A. Machine Learning Basics\
    A.1. Cross Validation \
    A.2. Motivating Example: Gradient Boosting Regressor \
B. Ridge and Lasso \
    B.1. Linear Regression \
    B.2. Ridge Regression \
    B.3. Lasso Regression \
Extend ML Basics:
    Random Forests
    Neural Networks
    Boosting

2. Heterogeneous Treatment Effects \
    2.1 Linear regression analysis \
    2.2 Sensitivity analysis \
    2.3 Comparing CATE model to ML prediction model
(Maybe get rid of this \ 
    3. Evaluating Causal Models \ 
    3.1 Random vs. non-random data \
    3.2 Sensitivity by model band \
    3.3 Cumulative Sensitivity \
    3.4 ROC for causal models )
4. Treatment effect estimators \
    4.1 From Outcomes to treatment effects \
    4.2 Continuous treatment effects \
5. Meta-Learners \
    5.1 S-learner \
    5.2 T-learner \
    5.3 X-learner \
6. Double ML \
    6.1 Recap: Frisch-Waugh-Lovell Theorem \
    6.2 Parametric Double ML ATE \
    6.3 Parametric Double ML CATE \
    6.4 Non-parametric Double ML CATE
7. Diff and Diff and ML
8. Sythetic Diff in Diff
9. Multi Armed Bandits (https://www.analyticsvidhya.com/blog/2023/02/solving-multi-arm-bandits-with-python/?utm_source=related_WP&utm_medium=https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/)



## Credits

This repository is based on the materials of:
- Mostly Harmless Econometrics (Angrist and Pischke, 2009)
- Causal Inference for the Brave and True (Matheus Facure Alves, 2022)
- Causal Analysis (Martin Huber, 2023)

It also relies heavily on the following papers:
- Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey. 2017. "Double/Debiased/Neyman Machine Learning of Treatment Effects." American Economic Review, 107 (5): 261-65.
- Susan Athey, Guido W. Imbens. August 2019. "Machine Learning Methods That Economists Should Know About" Annual Review of Economics  Vol. 11 Pages 685–725


