import pandas as pd
import quandl
import math
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

forecase_col = 'Adj. Close'
dataFrame.fillna(-99999, inplace = True) # Handling missing data(cant use NAN)

forecase_out = int(math.ceil(0.01*len(dataFrame))) # This means we'll predict 10% of data frame
dataFrame['label'] = dataFrame[forecase_col].shift(-forecase_out) # shifting the column
print(dataFrame.head())

# Model done, training left