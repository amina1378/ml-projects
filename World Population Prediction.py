#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import requests


# In[2]:


r = requests.get('https://www.worldometers.info/world-population/world-population-by-year/')


# In[3]:


data = pd.read_html(r.text)[0]
data = data[data['Year']>1950]


# In[4]:


data = data.set_index('Year')[['World Population','NetChange']]


# In[5]:


data = data.iloc[::-1]


# In[6]:


data


# In[7]:


data['World Population'].diff(1).diff(1).plot()


# In[9]:


data = data.drop('NetChange', axis=1)


# In[10]:


import pmdarima


# In[11]:


pmdarima.auto_arima(data['World Population'], seasonal=True, m=19)


# In[12]:


from statsmodels.tsa.arima.model import ARIMA


# In[13]:


model = ARIMA(data['World Population'], order=(0,2,0))


# In[14]:


results = model.fit()


# In[15]:


forc = results.get_forecast(30)


# In[16]:


pm = forc.predicted_mean
ci = forc.conf_int()


# In[17]:


pm.index = pm.index+1951
ci.index = ci.index+1951


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


plt.plot(data)
plt.plot(pm)
plt.fill_between(ci.index, ci['lower World Population'], ci['upper World Population'])


# In[ ]:




