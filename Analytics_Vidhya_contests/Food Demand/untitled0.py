# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:57:51 2020

@author: Madhumita Shankar
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score


df  = pd.read_csv("train.csv") 
X = df.iloc[:,[1,2,3,4,5,6,7]].values
y = df.iloc[:,-1].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#DTR
pipeline1 = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('DTR', DecisionTreeRegressor()),
])

pipeline1.fit(X_train,y_train)
y_pred = pipeline1.predict(X_test)
score = pipeline1.score(X_test,y_test)
#RF
pipeline2 = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('SVR', RandomForestRegressor(max_features='sqrt',criterion='mae')),
])

pipeline2.fit(X_train,y_train)
y_pred = pipeline2.predict(X_test)
score1 = pipeline2.score(X_test,y_test)

pipeline3 = Pipeline([
    ('Scaler', MinMaxScaler()),
    ('Poly', PolynomialFeatures(degree = 4)),
    ('LR', LinearRegression()),
])

pipeline3.fit(X_train,y_train)
y_pred = pipeline3.predict(X_test)
score2 = pipeline3.score(X_test,y_test)



df2 = pd.read_csv("test_QoiMO9B (1).csv")
df1 = pd.read_csv("sample_submission_hSlSoT6 (1).csv")

XX_test = df2.iloc[:,1:].values
yy_pred = pipeline2.predict(XX_test)
df1['num_orders'] = yy_pred
df1.to_csv("Submission.csv",index=False)

jobs = -1
param_range1 = [1,2,4,3,5,6,7,8,9,10]
param_range = [100,110,120,130]
grid_params_rf = [{
        'max_features': ["auto","sqrt","log2"],
        'min_samples_leaf': param_range1,
		'min_samples_split': param_range[1:]}]


gs_rf = GridSearchCV(estimator=pipeline2,
			param_grid=grid_params_rf,
			scoring='accuracy',
			cv=10, 
			n_jobs=jobs)

print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
grids = [gs_rf]

grid_dict = { 0: 'Random Forest'}
for idx, gs in enumerate(grids):
	print('\nEstimator: %s' % grid_dict[idx])	
	# Fit grid search	
	gs.fit(X_train, y_train)
	# Best params
	print('Best params: %s' % gs.best_params_)
	# Best training data accuracy
	print('Best training accuracy: %.3f' % gs.best_score_)
	# Predict on test data with best params
	y_pred = gs.predict(X_test)
	# Test data accuracy of model with best params
	print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
	# Track best (highest test accuracy) model
	if accuracy_score(y_test, y_pred) > best_acc:
		best_acc = accuracy_score(y_test, y_pred)
		best_gs = gs
		best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x_Train = pca.fit(X_train)
x_test = pca.fit_transform(X_test)
exp = pca.explained_variance_ratio_

pca1 = PCA(n_components=2)
XX


