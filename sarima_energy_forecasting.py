# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 08:30:52 2025

@author: avina
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.stattools import adfuller, kpss

df_ec = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Time Series Forecasting using Python/energyconsumption-201002-134452/energy consumption.csv')
df_ec.columns
df_ec.dtypes
df_ec.shape
df_ec.head()
df_ec.describe()


df_ec['DATE'] = pd.to_datetime(df_ec['DATE'], format="%m/%Y")
df_ec.set_index('DATE', inplace=True)

forecast_period = 36 # 36 months 
train = df_ec.iloc[:-forecast_period]  # Use all except the last 36 months for training
test = df_ec.iloc[-forecast_period:]

plt.figure(figsize=(12,8))

plt.plot(train.index, train['ENERGY_INDEX'], label='train_data')
plt.plot(test.index,test['ENERGY_INDEX'], label='test_data')
plt.legend(loc='best')
plt.title("Train and Test Data")
plt.show()

def adf_test(timeseries):
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(train['ENERGY_INDEX'])

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
kpss_test(train['ENERGY_INDEX'])

train['energyindex_log'] = np.log(train['ENERGY_INDEX'])
train['energyindex_log_diff'] = train['energyindex_log'] - train['energyindex_log'].shift(1)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace import sarimax


plt.figure(figsize=(12,6))
plt.subplot(211)
plot_acf(train['energyindex_log_diff'].dropna(), lags=25, ax=plt.gca())
plt.subplot(212)
plot_pacf(train['energyindex_log_diff'].dropna(), lags=15, ax=plt.gca())
plt.tight_layout()
plt.show()



# fit model
model = sarimax.SARIMAX(train['energyindex_log'], 
                        order=(2,1,1), 
                        seasonal_order=(1,1,1,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

results = model.fit()
print(results.summary())

# make predictions
pred = results.get_prediction(
    start=test.index[0],  # First date in test set
    end=test.index[-1],   # Last date in test set
    dynamic=False         
)

# Get predicted mean and confidence intervals
pred_mean = np.exp(pred.predicted_mean)  # Reverse log-transform
pred_ci = np.exp(pred.conf_int())        # Confidence intervals

test['SARIMA_Prediction'] = pred_mean

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['ENERGY_INDEX'], test['SARIMA_Prediction']))
print(f"RMSE: {rmse}")

# Calculate MAE
mae = mean_absolute_error(test['ENERGY_INDEX'], test['SARIMA_Prediction'])
print(f"MAE: {mae}")

plt.figure(figsize=(12,6))
plt.plot(train['ENERGY_INDEX'], label='Training Data')
plt.plot(test['ENERGY_INDEX'], label='Actual Test Data', color='blue')
plt.plot(test['SARIMA_Prediction'], label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(test.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='gray', alpha=0.2)
plt.title('SARIMA Forecast vs. Actual (Last 36 Months)')
plt.legend()
plt.show()

#trying other p,q values to test sarima fit
model2 = sarimax.SARIMAX(train['energyindex_log'], 
                        order=(1,1,1), 
                        seasonal_order=(1,1,1,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)

results2 = model2.fit()
print(results2.summary())

# make predictions
pred2 = results2.get_prediction(
    start=test.index[0],  # First date in test set
    end=test.index[-1],   # Last date in test set
    dynamic=False         
)

# Get predicted mean and confidence intervals
pred_mean2 = np.exp(pred2.predicted_mean)  # Reverse log-transform
pred_ci2 = np.exp(pred2.conf_int())        # Confidence intervals

test['SARIMA_Prediction_2'] = pred_mean2

# Calculate RMSE
rmse2 = np.sqrt(mean_squared_error(test['ENERGY_INDEX'], test['SARIMA_Prediction_2']))
print(f"RMSE: {rmse2}")

# Calculate MAE
mae2 = mean_absolute_error(test['ENERGY_INDEX'], test['SARIMA_Prediction_2'])
print(f"MAE: {mae2}")

plt.figure(figsize=(12,6))
plt.plot(train['ENERGY_INDEX'], label='Training Data')
plt.plot(test['ENERGY_INDEX'], label='Actual Test Data', color='blue')
plt.plot(test['SARIMA_Prediction_2'], label='SARIMA Forecast_2', color='red', linestyle='--')
plt.fill_between(test.index, pred_ci2.iloc[:,0], pred_ci2.iloc[:,1], color='gray', alpha=0.2)
plt.title('SARIMA Forecast_2 vs. Actual (Last 36 Months)')
plt.legend()
plt.show()