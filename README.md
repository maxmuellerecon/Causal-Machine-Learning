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
However, this repository is also a starting point for people interested in causal machine learning in python. \
The first half of the project is dedicated to machine learning in general. The second half focuses on causal machine learning.

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
&nbsp;&nbsp; I.1 Recap: Frisch-Waugh-Lovell Theorem \
&nbsp;&nbsp; I.2 Parametric Double ML ATE \
&nbsp;&nbsp; I.3 Parametric Double ML CATE \
&nbsp;&nbsp; I.4 Non-parametric Double ML CATE \
J. Diff and Diff and ML \
&nbsp;&nbsp; J.1 J.1 Two-way Fixed Effects \
&nbsp;&nbsp; J.2 Time Treatment Heterogeneity \
&nbsp;&nbsp; J.3 Flexible Functional Forms

## Libraries

In addition to the pre-installed libraries in the epp environment, the following libraries are used:

- keras
- tensorflow
- lightgbm
- scikit-learn
- graphviz
- linearmodels
- seaborn
- statsmodels

## Credits

This repository is based on the materials of:
- Mostly Harmless Econometrics (Angrist and Pischke, 2009)
- Causal Inference for the Brave and True (Matheus Facure Alves, 2022)
- Causal Analysis (Martin Huber, 2023)

It also relies on the following papers:
- Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey. 2017. "Double/Debiased/Neyman Machine Learning of Treatment Effects." American Economic Review, 107 (5): 261-65.
- Susan Athey, Guido W. Imbens. August 2019. "Machine Learning Methods That Economists Should Know About" Annual Review of Economics  Vol. 11 Pages 685–725
- Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 2019. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the National Academy of Sciences, 116 (10): 4156-4165.
- Brantly Callaway, Pedro H.C. Sant’Anna. 2021. "Difference-in-Differences with multiple time periods" Journal of Econometrics. Volume 225, Issue 2.
- Andrew Goodman-Bacon. 2021 "Difference-in-differences with variation in treatment timing" Journal of Econometrics. Volume 225, Issue 2.
