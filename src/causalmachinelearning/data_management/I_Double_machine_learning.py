#I_Double-machine learning

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
#from nb21 import cumulative_gain, elast
import statsmodels.formula.api as smf
from matplotlib import style
style.use("ggplot")
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict
from config import BLDFIG, DATA, BLDTBL


this_dir = Path('.')
this_dir = this_dir.resolve()
test = pd.read_csv(DATA / "ice_cream_sales_rnd.csv")
train = pd.read_csv(DATA / "ice_cream_sales.csv")


#Positives: Works for continuous and binary treatments and has rigorous validity

#####################6.1 Recap: Frisch-Waugh-Lovell Theorem########################
def plot_pattern(df, x_var, y_var):
    np.random.seed(123)
    sns.scatterplot(data=df.sample(1000), x=x_var, y=y_var, hue="weekday")
    plt.savefig(os.path.join(BLDFIG, "06_Scatterplot_Sales_Price.png"))
    return plt.show()

#Observe that prices on weekends are higher, want to remove that form of bias
plot_pattern(train, "price", "sales")


#FWL Theorem
#1.) Regress Outcome (Sales) on Covariates (Weekday, Temp, Cost)
my = smf.ols("sales~temp+C(weekday)+cost", data=train).fit()
#2.) Regress Treatment (Price) on Covariates (Weekday, Temp, Cost)
mt = smf.ols("price~temp+C(weekday)+cost", data=train).fit()
#3.) Obtain the residuals from the 2 regressions and regress y - y_hat on t - t_hat
table1 = smf.ols("sales_res~price_res", 
            data=train.assign(sales_res=my.resid, # sales residuals
                          price_res=mt.resid) # price residuals
            ).fit().summary()

# Export to LaTeX
latex_code = table1.as_latex()
print(latex_code)
with open(BLDTBL / '05_1_FWL.tex', 'w') as f:
    f.write(latex_code)

#Verify that FWL is the same as usual regression:
smf.ols("sales~price+temp+C(weekday)+cost", data=train).fit().params["price"]


#####################6.2 Parametric Double ML ATE############################################
#Key Idea: use ML model to construct the residual functions, so no OLS, hence capture non-linearities, interactions 
#between X and T and X and Y

#1.) Regress Outcome (Sales) on Covariates (Weekday, Temp, Cost) with flexible ML model
#2.) Regress Treatment (Price) on Covariates (Weekday, Temp, Cost) with flexible ML model
#3.) Obtain the residuals from the 2 regressions
#4.) Regress y - my_hat on t - mt_hat, get causal effect, estimate by e.g. OLS

#Positive: Gain flexibility
#Negative: Could overfit in steps 1 and 2

#Solution to overfitting: Use cross-fitting, where we fit model on one part of data and make residuals and predictions in other


#1.) Debias the treatment +3.) obtain residuals for t
Y = "sales"
T = "price"
X = ["temp", "weekday", "cost"]
debias_m = LGBMRegressor(max_depth=3)
#Estimate residuals: t_tilde = t - t_hat(from ML model)
#add train[T].mean for visualization
train_pred = train.assign(price_res =  train[T] -
                          cross_val_predict(debias_m, train[X], train[T], cv=5)
                          + train[T].mean()) 

#Visualize
def plot_debiased(df):
    np.random.seed(123)
    sns.scatterplot(data=df.sample(1000), x="price_res", y="sales", hue="weekday");
    plt.savefig(os.path.join(BLDFIG, "06_Scatterplot_Debiased.png"))
    return plt.show()
plot_debiased(train_pred)

#2.) Reduce the variance from y +3.) obtain residuals for Y
denoise_m = LGBMRegressor(max_depth=3)
#Estimate residuals: y_tilde = y - y_hat(from ML model)
#add train[Y].mean for visualization
train_pred = train_pred.assign(sales_res =  train[Y] -
                               cross_val_predict(denoise_m, train[X], train[Y], cv=5)
                               + train[Y].mean())

#Visualize
def plot_debiase_denoised(df):
    np.random.seed(123)
    sns.scatterplot(data=df.sample(1000), x="price_res", y="sales_res", hue="weekday");
    plt.savefig(os.path.join(BLDFIG, "06_Scatterplot_Debiased_Denoised.png"))
    return plt.show()
plot_debiase_denoised(train_pred)

#4.) Use the orthogonalized residuals to estimate the causal effect (ATE), get negative relationship
final_model = smf.ols("sales_res~price_res", data=train_pred).fit()
final_model.summary()

#In coparison to unorthogonalized model, have positive relationship
basic_model = smf.ols("sales~price", data=train).fit()
basic_model.summary()


#####################6.3 Parametric Double ML CATE############################################
#Now we want to produce the CATE, not the ATE, interact the residuals with the covariates
final_model_cate = smf.ols(formula='sales_res ~ price_res * (temp + C(weekday) + cost)', data=train_pred).fit()

#Use that model to predict the CATE
#Which is also called the R-Learner (Residualization)
cate_test = test.assign(cate=final_model_cate.predict(test.assign(price_res=1))
                        - final_model_cate.predict(test.assign(price_res=0)))


#####################6.4 Non-parametric Double ML CATE############################################
#We stil had linear specifications in the last step, now we want to use non-linear models
#Same as before:
debias_m = LGBMRegressor(max_depth=3)
denoise_m = LGBMRegressor(max_depth=3)
train_pred = train.assign(price_res =  train[T] - cross_val_predict(debias_m, train[X], train[T], cv=5),
                          sales_res =  train[Y] - cross_val_predict(denoise_m, train[X], train[Y], cv=5))

#Define a causal loss function, that the R-learner minimizes
model_final = LGBMRegressor(max_depth=3)
# create the weights
w = train_pred["price_res"] ** 2
# create the transformed target
y_star = (train_pred["sales_res"] / train_pred["price_res"])
# use a weighted regression ML model to predict the target with the weights.
model_final.fit(X=train[X], y=y_star, sample_weight=w)
#Estimate the treatment effect
cate_test_non_param = test.assign(cate=model_final.predict(test[X]))
