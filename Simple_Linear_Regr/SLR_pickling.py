import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import model_selection, preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle # For data serialization

style.use('ggplot')

dataFrame = quandl.get('WIKI/GOOGL'); #The following is the Google's Stock data
dataFrame = dataFrame[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
dataFrame['HL_PCT'] = (dataFrame['Adj. High'] - dataFrame['Adj. Close'])/dataFrame['Adj. Close']*100
dataFrame['PCT_Change'] = (dataFrame['Adj. Close'] - dataFrame['Adj. High'])/dataFrame['Adj. High']*100
dataFrame = dataFrame[['Adj. Close' ,'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
dataFrame.fillna(-99999, inplace = True) # Handling missing data(cant use NAN)

forecast_out = int(math.ceil(0.01*len(dataFrame))) # This means we'll predict 10% of data frame
dataFrame['label'] = dataFrame[forecast_col].shift(-forecast_out) # shifting the column
print(dataFrame.head())

x = np.array(dataFrame.drop(['label'], 1)) # Everything except label
x = preprocessing.scale(x)
x_lately = x[-forecast_out:] #for prediction
x = x[:-forecast_out:]

dataFrame.dropna(inplace = True)
y = np.array(dataFrame['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.2)

classifier = LinearRegression(n_jobs = 10) # To do parallelism and increase speed (Super Threading) we do n_jobs and number of process to run

classifier.fit(x_train, y_train)
'''
This is best place to save your model(classifier) as we have trained it
and its in ready to go phase, further we can train it once in month
'''
with open('linearregression.pickle', 'wb') as f:
	pickle.dump(classifier, f);
# To use the classifier in other files use these 2 commands
# pickle_in = open('linearregression.pickle', 'rb')
# classifier = pickle.load(pickle_in)

accuracy = classifier.score(x_test, y_test)

print(accuracy)
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