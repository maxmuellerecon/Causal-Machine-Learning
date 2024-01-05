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

The class is structured as follows: 

##### Part 1: Introduction to Machine Learning 
 
A. Machine Learning Basics \
&nbsp;&nbsp; A.1. Cross Validation \
&nbsp;&nbsp; A.2. Motivating Example: Gradient Boosting Regressor \
B. Ridge and Lasso \
&nbsp;&nbsp; B.1. Linear Regression \
&nbsp;&nbsp; B.2. Ridge Regression \
&nbsp;&nbsp; B.3. Lasso Regression \
C. Neural Networks \
D. Decision Trees \
E. Ensemble Learning \
&nbsp;&nbsp; E.1 Bagging \
&nbsp;&nbsp; E.2 Boosting


##### Part 2: Causal Machine Learning 

F. Heterogeneous Treatment Effects \
&nbsp;&nbsp; F.1 Linear regression analysis \
&nbsp;&nbsp; F.2 Sensitivity analysis \
&nbsp;&nbsp; F.3 Comparing CATE model to ML prediction model \
G. Treatment effect estimators \
&nbsp;&nbsp; G.1 From Outcomes to treatment effects \
&nbsp;&nbsp; G.2 Continuous treatment effects \
H. Meta-Learners \
&nbsp;&nbsp; H.1 S-learner \
&nbsp;&nbsp; H.2 T-learner \
&nbsp;&nbsp; H.3 X-learner \
I. Double ML \
&nbsp;&nbsp; 6.1 Recap: Frisch-Waugh-Lovell Theorem \
&nbsp;&nbsp; 6.2 Parametric Double ML ATE \
&nbsp;&nbsp; 6.3 Parametric Double ML CATE \
&nbsp;&nbsp; 6.4 Non-parametric Double ML CATE \
J. Diff and Diff and ML \
K. Sythetic Diff in Diff \
9. Multi Armed Bandits (https://www.analyticsvidhya.com/blog/2023/02/solving-multi-arm-bandits-with-python/?utm_source=related_WP&utm_medium=https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/)



## Credits

This repository is based on the materials of:
- Mostly Harmless Econometrics (Angrist and Pischke, 2009)
- Causal Inference for the Brave and True (Matheus Facure Alves, 2022)
- Causal Analysis (Martin Huber, 2023)

It also relies on the following papers:
- Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey. 2017. "Double/Debiased/Neyman Machine Learning of Treatment Effects." American Economic Review, 107 (5): 261-65.
- Susan Athey, Guido W. Imbens. August 2019. "Machine Learning Methods That Economists Should Know About" Annual Review of Economics  Vol. 11 Pages 685–725


