import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

quandl.ApiConfig.api_key = "hcm2ckbK5zV5FMW7zzsy"

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Volume', 'Adj. Close']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.00
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.00

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

# Reset index of df from Date to Integers.
#df.reset_index(inplace=True)

# Forecasting Output Column
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# Forecasting output price after time 
forecast_out = 5

df['label'] = df[forecast_col].shift(-forecast_out)
#print(df.head())

print(len(df))
 
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = X[:-forecast_out]
y = y[:-forecast_out]
X_predict = X[-forecast_out:]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

'''
with open('LinearReg_StockPrice.pickle', 'wb') as f:
    pickle.dump(clf, f)


pickle_in = open('LinearReg_StockPrice.pickle', 'rb')
clf = pickle.load(pickle_in)
'''

accuracy = clf.score(X_test, y_test)
print(accuracy) 
y_predict = clf.predict(X_predict)
print(y_predict)
