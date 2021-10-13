#!/usr/bin/env python
# coding: utf-8

# # **Application of Monte Carlo Simulation - Tesla**

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from matplotlib import pyplot as plt


# In[2]:


ticker = 'TSLA' # ticker
intervals = 30 # time steps forecasted into future
iterations = 25 # amount of simulations


# In[3]:


tesla = yf.Ticker(ticker)


# In[4]:


df = tesla.history(period = "max")


# In[5]:


df.head(5)


# In[6]:


df.tail(5)


# In[7]:


df = df.rename(columns={"Close": ticker})
df = df[['TSLA']]
df.head()


# In[8]:


#Take Log
log_df = np.log(1 + df.pct_change())


# In[9]:


#Plot of asset historical closing price
df.plot(figsize=(10, 6));


# In[10]:


#Plot of log returns
log_df.plot(figsize = (10, 6))


# In[11]:


#Setting up drift and random component in relation to asset data
u = log_df.mean()
var = log_df.var()
drift = u - (0.5 * var)
stdv = log_df.std()


# In[12]:


returns = np.exp(drift.values + stdv.values * norm.ppf(np.random.rand(intervals, iterations)))


# In[13]:


#Takes last data point as startpoint point for simulation
S_zero = df.iloc[-1]
list_pred = np.zeros_like(returns)
list_pred[0] = S_zero


# In[14]:


#Applies Monte Carlo simulation in asset
for t in range(1, intervals):
    list_pred[t] = list_pred[t - 1] * returns[t]


# In[15]:


#Plot simulations
plt.figure(figsize=(10,6))
plt.plot(list_pred);


# ### * Large volatility
# ### * From today's price of 800, can swing from 500 to 1100
