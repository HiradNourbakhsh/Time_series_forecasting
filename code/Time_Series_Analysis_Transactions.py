#!/usr/bin/env python
# coding: utf-8

# # Here, we will build a time series forecasting model to predict grocery store transactions.

# In[85]:


import warnings
import math
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[46]:


transactions = pd.read_csv('/Users/hiradnourbakhsh/Desktop/INSY 695/Group Assignment/favorita-grocery-sales-forecasting/transactions.csv')


# In[47]:


transactions


# In[48]:


transactions = transactions.drop(columns = ['store_nbr'])


# In[49]:


transactions


# In[50]:


import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[51]:


transactions.index


# In[52]:


transactions = transactions.groupby('date')['transactions'].sum().reset_index()


# In[53]:


transactions


# In[54]:


#btc.index = pd.to_datetime(btc['Date'], format = '%Y-%m-%d')

transactions.index = pd.to_datetime(transactions['date'], format = '%Y-%m-%d')


# In[55]:


transactions.index


# In[56]:


del transactions['date']


# In[57]:


transactions


# In[58]:


y = transactions['transactions'].resample('MS').mean()


# In[59]:


y


# In[60]:


y.isnull().sum()


# # Checking for seasonality

# In[61]:


y.plot(figsize = (15,6))
plt.show()


# Note the obvious seasonality (huge spike in transactions towards the end of the year) and the overall increasing trend.

# In[62]:


from pylab import rcParams
rcParams['figure.figsize'] = 11,9

decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')
fig = decomposition.plot()
fig.show()


# # Parameter Selection

# In[88]:


p = d = q = range(0,2)
pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d,q))]

print('Potential paramater combinations:')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[64]:


warnings.filterwarnings('ignore')

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order = param, seasonal_order = param_seasonal,
                                           enforce_stationarity = False, enforce_invertibility = False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[1]:


# use the following combination (lowest AIC of 516):
# ARIMA(0, 1, 1)x(1, 1, 1, 12)12 - AIC:516.061805070915


# In[66]:


mod = sm.tsa.statespace.SARIMAX(y, 
                                order = (0,1,1), 
                                seasonal_order = (1,1,1,12), 
                                enforce_stationarity = False, 
                               enforce_invertibility = False)
results = mod.fit()
print(results.summary().tables[0])
print(results.summary().tables[1])


# In[67]:


# could get rid of ma.L1, ma.S.L12 (insignificant to model performance)


# In[68]:


results.plot_diagnostics(figsize = (15,12))
plt.show()


# In[69]:


# residuals are normally distributed


# # Validating Forecasts

# In[70]:


pred = results.get_prediction(start = pd.to_datetime('2017-01-01'), dynamic = False)
pred_ci = pred.conf_int()


# In[71]:


ax = y.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Transactions')
plt.legend()

plt.show()


# Our forecasts align well with the actual values.

# # Testing our model's performance

# In[87]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

mse = ((y_forecasted - y_truth)**2).mean()
print('MSE = ', mse)
print('root MSE = ', math.sqrt(mse))


# In[80]:


# MAPE

import numpy as np

def mape(actual_value, predicted_value):
    actual_value, predicted_value = np.array(actual_value), np.array(predicted_value)
    return np.mean(np.abs((actual_value - predicted_value) / actual_value)) * 100


# In[89]:


mape(y_truth, y_forecasted)


# In[94]:


(abs(y_truth - y_forecasted)/(y_truth) * 100).mean()


# In[95]:


y_truth - y_forecasted


# In[98]:


pd.DataFrame([y_truth,y_forecasted]).T.to_clipboard()


# In[99]:


pd.DataFrame([y_truth,y_forecasted]).T


# # Our model performs well, with a MAPE of 0.96%.

# In[ ]:




