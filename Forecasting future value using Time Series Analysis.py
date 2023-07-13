import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
import datetime
import sys
import seaborn as sns
import statsmodels
import statsmodels.stats.diagnostic as diag
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy.stats.mstats import normaltest
from matplotlib.pyplot import acorr
get_ipython().magic('matplotlib inline')
import quandl

APPLE = quandl.get("YAHOO/AAPL")
APPLE.to_csv('APPLE.csv')
lastAPPLE = APPLE[-1:]
APPLE = APPLE[:-1]
APPLE.head()

# Lets take Log of the daily the closing value to ensures that level induced volatility does not interfere with the stationarity of series.

APPLE['date']= APPLE.index
APPLE['APPLE']= APPLE.Close
APPLE['logAPPLE']= np.log(APPLE['APPLE'])
APPLE['diflogAPPLE'] = APPLE['logAPPLE'] - APPLE['logAPPLE'].shift(periods=-1)
APPLE = APPLE.dropna()

# As we are interested in the day to day change of the stock prices, we take the difference of the closing values. This differencing is another way to get the time series to be stationary.

fig, ax = plt.subplots(figsize=(12,8))

plt.subplot(2, 1, 1)
plt.plot(APPLE.date, APPLE.APPLE, label = "Log of APPLE Closing Price")
plt.title("Level APPLE Closing Price", size = 20,)
plt.ylabel("Price in Dollars", size = 10)

plt.subplot(2, 1, 2)
plt.plot(APPLE.date, APPLE.diflogAPPLE, label = '1st Diffrence of Log of APPLE', color = 'g')
plt.title("Difference between Log of APPLE Closing Price", size = 10,)
plt.ylabel("Difference between APPLE Closing Price", size = 10)
plt.xlabel('Month', size = 10)

print ('Results of Dickey-Fuller Test:')
dftest = adfuller(APPLE.diflogAPPLE, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

ararray = (APPLE.logAPPLE.dropna().as_matrix())
p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(ararray, (p,d,q)).fit()
            x = model.aic
            x1 = (p,d,q)
            print (x1, x)
            aic.append(x)
            pdq.append(x1)
        except:
            pass
            
keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))
ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()

numofsteps = 1
stepahead = ARIMIAmod.forecast(numofsteps)[0]
ferrors = ARIMIAmod.forecast(numofsteps)[2]
ferrors
print ('%s Steps Ahead Forecast Value is:' % numofsteps, np.exp(stepahead))
print ('%s Steps Ahead 95 percent CI is:' % numofsteps, np.exp(ferrors[0]))
print ('April 26th 2016 Close (most recent): %s ' % lastAPPLE.Close)

