#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:32:18 2019

@author: jetfire
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.style.use('fivethirtyeight')
#Data Preprocessing.
df1 = pd.read_csv('Test_0qrQsBZ.csv')
df = pd.read_csv('Train_SU63ISt.csv')
train = df[0:10392]
test = df[10392:]
df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M')
df.index = df.Timestamp 
df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M')
train.index = train.Timestamp 
train = train.resample('D').mean()
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M')
test.index = test.Timestamp 
test = test.resample('D').mean()
train.Count.plot()
test.Count.plot()
plt.show()

import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize']= 18,8
decomp = sm.tsa.seasonal_decompose(train,model="additive")
fig = decomp.plot()
plt.xlabel("Datetime")
plt.savefig("pic.jpg")
plt.show()

import itertools
a=b=c = range(0,2)
pdq = list(itertools.product(a,b,c))
seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(a,b,c))]
for parm in pdq:
    for parm_sea in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train.Count,
                                            order=parm,
                                            seasonal_order=parm_sea,
                                            enforce_invertibility=False,
                                            enforce_stationarity=False)
            result = mod.fit()
            print('ARIMA{}x{}12-AIC:{}'.format(parm,parm_sea,result.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(train.Count,order=(1,1,1),seasonal_order=(1,1,1,12),
                                enforce_invertibility=False,enforce_stationarity=False)
results = mod.fit()
print(results.summary())
results.plot_diagnostics()
plt.savefig("dia.png")
plt.show()
y_hat_ = test.copy()
pred = results.get_prediction(start = pd.to_datetime('2013-05-01'),dynamic=False)
pred_cli = pred.conf_int()
y_forcasted = pred.predicted_mean
y_truth = train.Count['2013-05-01':]
mse = ((y_forcasted-y_truth)** 2).mean()
rmse = np.sqrt(mse)
df1['forcasted'] = results.get_forecast('2015-4-26')
sm.tsa.statespace.SARIMAX.prepare_dataSARIMAXResults()


      
