# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:02:20 2020

@author: ayushjain9
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

# Choose Relevant columns
# Get dummy variable . Every categorical col should have each col
# train test split
# multiple linear regression
# lasso regression
# random forest
# tune model usiing GridSearchCV
# test ensembles


# Choose relevant Columns
df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership',
               'Industry','Sector','Revenue','num_comp','hourly',
               'employer_provided', 'job_state','same_state',
               'age','python_yn','spark','aws','excel','job_simp',
               'seniority','desc_len']]


# Get dummy variable . Every categorical col should have each col
df_dum = pd.get_dummies(df_model)


# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values # array is recommended. that's why values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)


# multiple linear regression
# stats ols model is used for analysing

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()
# p< 0.05 is relevant column
    
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
# cross validation will run sample and vaildation set. Mini test

lm = LinearRegression()
lm.fit(X_train, y_train)
cross_val_score(lm,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 2)
np.mean(cross_val_score(lm,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 2))
""" linear regression is bit giving arounf 22000$ of change from mean"""

lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

""" lasso is giving 19 % error where as linear regression was giving 22%"""

# random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))
""" giving 14 %""" # Tuning needed

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_ 
#14.8

gs.best_estimator_
# to check the parametres

# test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

# 14% error

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])








