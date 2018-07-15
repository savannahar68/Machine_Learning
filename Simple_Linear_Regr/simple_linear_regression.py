import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import model_selection, preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
'''
Quandl library Gets financial data directly
We'll use Quandl for getting our dataset
'''
dataFrame = quandl.get('WIKI/GOOGL'); #The following is the Google's Stock data
# print(dataFrame.head()) # gives you all the features 
# print(dataFrame)
dataFrame = dataFrame[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

# Creating a extra feature column from High and Low (High Low Percentage)
dataFrame['HL_PCT'] = (dataFrame['Adj. High'] - dataFrame['Adj. Close'])/dataFrame['Adj. Close']*100

dataFrame['PCT_Change'] = (dataFrame['Adj. Close'] - dataFrame['Adj. High'])/dataFrame['Adj. High']*100

# Making final dataframe of the only columns which we want
dataFrame = dataFrame[['Adj. Close' ,'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
dataFrame.fillna(-99999, inplace = True) # Handling missing data(cant use NAN)

forecast_out = int(math.ceil(0.01*len(dataFrame))) # This means we'll predict 10% of data frame
dataFrame['label'] = dataFrame[forecast_col].shift(-forecast_out) # shifting the column
print(dataFrame.head())

# Real thing start

x = np.array(dataFrame.drop(['label'], 1)) # Everything except label
x = preprocessing.scale(x)
x_lately = x[-forecast_out:] #for prediction
x = x[:-forecast_out:]

dataFrame.dropna(inplace = True)
y = np.array(dataFrame['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.2)

classifier = LinearRegression(n_jobs = 10) # To do parallelism and increase speed (Super Threading) we do n_jobs and number of process to run
#							  n_jobs = -1 this will try to run as many jobs as possible	
'''
To try with SVM we will change only 1 line
classifier = svm.SVR(kernel = 'poly') 
It gives pretty bad accuracy
'''
classifier.fit(x_train, y_train)
accuracy = classifier.score(x_test, y_test)

print(accuracy)

# Testing training done now forecasting and predicting 
forecast_set = classifier.predict(x_lately)
print(forecast_set, accuracy, forecast_out)

# Plotting

dataFrame['forecast'] = np.nan
last_date= dataFrame.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #seconds a day
next_unix = last_unix + one_day

#For calculating date for x axis
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	dataFrame.loc[next_date] = [np.nan for _ in range(len(dataFrame.columns)-1)] + [i]

dataFrame['Adj. Close'].plot()
dataFrame['forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()	