#!/usr/bin/env python
# coding: utf-8

# In[40]:


import sys 
import pickle
from datetime import datetime, timedelta
from itertools import product

import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from pandas.plotting import register_matplotlib_converters
import pmdarima as pm

register_matplotlib_converters()

import math
from math import sqrt 
from numpy import inf

import warnings
warnings.filterwarnings("ignore")

import statistics 
import random

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from math import sqrt 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("mock_kaggle.csv")


# In[3]:


dataset.head()


# In[14]:


source = dataset
source = source.sort_values('data')
df_target = source.groupby(["data", 'estoque', 'preco'])['venda'].sum().to_frame().reset_index()
min_date = str(source.data.min())[:10]
df_target['week'] = source.data.apply(lambda x: (datetime.strptime(str(x)[0:10], '%Y-%m-%d') - datetime.strptime(min_date, "%Y-%m-%d")).days // 7)
df_target = df_target.groupby(["data", 'venda', 'preco', 'week'])['estoque'].sum().to_frame().reset_index()
df_target = df_target.groupby(["data", 'venda', 'estoque', 'week'])['preco'].sum().to_frame().reset_index()
df_target = df_target.drop(['data'], axis = 1)


# In[16]:


df_target.head()


# In[18]:


list_retail_store = ['venda', 'estoque', 'preco']


# In[19]:


df_target.week.max()


# ### For Venda Retail Store

# ## ARIMA

# In[76]:


df_used = df_target.drop(['estoque', 'preco'], axis = 1)
list_retail_store = ['venda']
df_used.head()


# In[92]:


get_ipython().run_cell_magic('time', '', "\ny_true_final_arima = []\ny_pred_final_arima = []\nz = 1\n\nx=0\nwindow = 15\nstart_week = 132\nend_week = 133\ntarget_week = 134\nx_y = 0\n\nfor i in list_retail_store:\n    df = df_used\n    x_y = 0\n\n    for x in range(1):\n        df = df.loc[(df.week >= start_week) & (df.week <= end_week)] \n        df_true = df.loc[(df.week == target_week)]            \n        j = 0\n        y_true_sampling = []\n        list_x = []\n            \n        data_test = df\n        data_test.index = data_test['week']\n        test_array = data_test.to_numpy()\n            \n        from sklearn.preprocessing import StandardScaler\n        scaler = StandardScaler()\n            \n        df_x = list(data_test.venda.values)\n        \n        if(len(df_x) < 1):\n            df_x = [0] * 30\n        else:\n            df_x = df_x\n               \n\n        sklearn = df_x\n        reshape = np.reshape(sklearn, (-1,1))\n        scaler.fit(reshape)\n            \n        reshape_final = scaler.transform(reshape)\n        reshape_final_2 = np.reshape(reshape_final, (-1))\n\n        lists_final_2 = list(reshape_final_2)\n        min_list_value = min(lists_final_2)\n\n        for t in range(1, 31):\n            if(lists_final_2.index == t):\n                list_x.append(lists_final_2[t])\n            else: \n                list_x.append(min_list_value)\n\n        model = pm.auto_arima(list_x, start_p=0, start_q=0,\n                  test='adf',   \n                  max_p=1, max_q=1, \n                  m=1,  \n                  d=1,   \n                  seasonal=False,   \n                  start_P=1, \n                  D=1, \n                  trace=False,\n                  error_action='ignore',  \n                  suppress_warnings=True, \n                  stepwise=True)\n        n_periods = 1\n\n        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)\n        index_of_fc = np.arange(len(lists_final_2), len(lists_final_2)+n_periods)\n        fc_series = pd.Series(fc, index=index_of_fc)\n\n        fc_series_inverse = scaler.inverse_transform(fc_series.values)\n        list_y_pred = list(np.round(fc_series_inverse, 0))\n\n        list_y_pred = [0 if i < 0 else i for i in list_y_pred]\n\n        y_true = list(df_true.venda.values)\n\n        if(1 > len(y_true)):\n            y_true_sampling = y_true\n            for j in range(0, (1-len(y_true))):\n                y_true_sampling.append(0)\n        else:\n            for j in range(1):\n                y_true_sampling.append(y_true[j])\n\n        y_true_final_arima += y_true_sampling\n        y_pred_final_arima += list_y_pred\n\n        start_week += 1\n        end_week += 1\n        target_week += 1\n        x_y += 1\n        \n    start_week = 132\n    end_week = 133\n    target_week = 134    \n        \n    print(z, i)\n    z += 1")


# In[93]:


x_pred_arima = y_pred_final_arima
pred_without_nan_arima = [0 if math.isnan(x) else x for x in x_pred_arima]


# In[94]:


#count_data
mae = mean_absolute_error(y_true_final_arima, pred_without_nan_arima)
mse = mean_squared_error(y_true_final_arima, pred_without_nan_arima)
rms = sqrt(mse)

print("MAE : ", mae)
print("MSE : ", mse)
print("RMSE : ", rms)


# ## Croston

# In[97]:


get_ipython().run_cell_magic('time', '', "\nextra_periods=1\nalpha= 0.6\n\ny_true_final = []\ny_pred_final = []\n\nz = 1\n\nx=0\nstart_week = 132\nend_week = 133\ntarget_week = 134\n\nfor i in list_retail_store:\n    df_warehouse = df_used\n        \n    for x in range(1):\n        df = df_warehouse.loc[(df_warehouse.week >= start_week) & (df_warehouse.week <= end_week)] \n        df_true = df_warehouse.loc[(df_warehouse.week == target_week)]\n\n        y_true_sampling = []\n        j = 0\n\n        qty = df.loc[:,'venda'].as_matrix()\n        my_new_list = [i * 1 for i in qty]\n\n        if(len(my_new_list) < 1):\n            my_new_list = [0] * 2\n        else:\n            my_new_list = my_new_list\n\n        d = np.array(my_new_list) \n        cols = len(d) \n        d = np.append(d,[np.nan]*extra_periods) \n\n        a,p,f = np.full((3,cols+extra_periods),np.nan)\n        q = 1 \n\n        first_occurence = np.argmax(d[:cols]>0)\n        a[0] = d[first_occurence]\n        p[0] = 1 + first_occurence\n        f[0] = a[0]/p[0]\n\n        for t in range(0,cols):        \n            if d[t] > 0:\n                a[t+1] = alpha*d[t] + (1-alpha)*a[t] \n                p[t+1] = alpha*q + (1-alpha)*p[t]\n                f[t+1] = a[t+1]/p[t+1]\n                q = 1           \n            else:\n                a[t+1] = a[t]\n                p[t+1] = p[t]\n                f[t+1] = f[t]\n                q += 1\n\n        a[cols+1:cols+extra_periods] = a[cols]\n        p[cols+1:cols+extra_periods] = p[cols]\n        f[cols+1:cols+extra_periods] = f[cols]\n\n        y_pred_list = list(f)\n        y_pred = y_pred_list[-1]\n\n        y_true = list(df_true.venda.values)\n\n        if(1 > len(y_true)):\n            y_true_sampling = y_true\n        for j in range(0, (1-len(y_true))):\n            y_true_sampling.append(0)\n        else:\n            for j in range(0, 1):\n                y_true_sampling.append(y_true[j])\n\n        y_true_final += y_true_sampling\n        y_pred_final.append(round(y_pred_list[-1]))\n\n        start_week += 1\n        end_week += 1\n        target_week += 1\n        \n    start_week = 132\n    end_week = 133\n    target_week = 134\n    print(z, i)\n    z += 1\n    \nx_pred_arima = y_pred_final_arima\npred_without_nan_arima = [0 if math.isnan(x) else x for x in x_pred_arima]")


# In[99]:


#count_data
mae = mean_absolute_error(y_true_final_arima, pred_without_nan_arima)
mse = mean_squared_error(y_true_final_arima, pred_without_nan_arima)
rms = sqrt(mse)

print("MAE : ", mae)
print("MSE : ", mse)
print("RMSE : ", rms)


# In[ ]:





# ### For estoque retail store

# In[100]:


df_used = df_target.drop(['venda', 'preco'], axis = 1)
list_retail_store = ['estoque']
df_used.head()


# ## ARIMA

# In[108]:


get_ipython().run_cell_magic('time', '', "\ny_true_final_arima = []\ny_pred_final_arima = []\nz = 1\n\nx=0\nwindow = 15\nstart_week = 132\nend_week = 133\ntarget_week = 134\nx_y = 0\n\nfor i in list_retail_store:\n    df = df_used\n    x_y = 0\n\n    for x in range(1):\n        df = df.loc[(df.week >= start_week) & (df.week <= end_week)] \n        df_true = df.loc[(df.week == target_week)]            \n        j = 0\n        y_true_sampling = []\n        list_x = []\n            \n        data_test = df\n        data_test.index = data_test['week']\n        test_array = data_test.to_numpy()\n            \n        from sklearn.preprocessing import StandardScaler\n        scaler = StandardScaler()\n            \n        df_x = list(data_test.estoque.values)\n        \n        if(len(df_x) < 1):\n            df_x = [0] * 30\n        else:\n            df_x = df_x\n               \n\n        sklearn = df_x\n        reshape = np.reshape(sklearn, (-1,1))\n        scaler.fit(reshape)\n            \n        reshape_final = scaler.transform(reshape)\n        reshape_final_2 = np.reshape(reshape_final, (-1))\n\n        lists_final_2 = list(reshape_final_2)\n        min_list_value = min(lists_final_2)\n\n        for t in range(1, 31):\n            if(lists_final_2.index == t):\n                list_x.append(lists_final_2[t])\n            else: \n                list_x.append(min_list_value)\n\n        model = pm.auto_arima(list_x, start_p=0, start_q=0,\n                  test='adf',   \n                  max_p=1, max_q=1, \n                  m=1,  \n                  d=1,   \n                  seasonal=False,   \n                  start_P=1, \n                  D=1, \n                  trace=False,\n                  error_action='ignore',  \n                  suppress_warnings=True, \n                  stepwise=True)\n        n_periods = 1\n\n        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)\n        index_of_fc = np.arange(len(lists_final_2), len(lists_final_2)+n_periods)\n        fc_series = pd.Series(fc, index=index_of_fc)\n\n        fc_series_inverse = scaler.inverse_transform(fc_series.values)\n        list_y_pred = list(np.round(fc_series_inverse, 0))\n\n        list_y_pred = [0 if i < 0 else i for i in list_y_pred]\n\n        y_true = list(df_true.estoque.values)\n\n        if(1 > len(y_true)):\n            y_true_sampling = y_true\n            for j in range(0, (1-len(y_true))):\n                y_true_sampling.append(0)\n        else:\n            for j in range(1):\n                y_true_sampling.append(y_true[j])\n\n        y_true_final_arima += y_true_sampling\n        y_pred_final_arima += list_y_pred\n\n        start_week += 1\n        end_week += 1\n        target_week += 1\n        x_y += 1\n        \n    start_week = 132\n    end_week = 133\n    target_week = 134    \n        \n    print(z, i)\n    z += 1\n    \nx_pred_arima = y_pred_final_arima\npred_without_nan_arima = [0 if math.isnan(x) else x for x in x_pred_arima]")


# In[109]:


#count_data
mae = mean_absolute_error(y_true_final_arima, pred_without_nan_arima)
mse = mean_squared_error(y_true_final_arima, pred_without_nan_arima)
rms = sqrt(mse)

print("MAE : ", mae)
print("MSE : ", mse)
print("RMSE : ", rms)


# ## Croston

# In[110]:


get_ipython().run_cell_magic('time', '', "\nextra_periods=1\nalpha= 0.6\n\ny_true_final = []\ny_pred_final = []\n\nz = 1\n\nx=0\nstart_week = 132\nend_week = 133\ntarget_week = 134\n\nfor i in list_retail_store:\n    df_warehouse = df_used\n        \n    for x in range(1):\n        df = df_warehouse.loc[(df_warehouse.week >= start_week) & (df_warehouse.week <= end_week)] \n        df_true = df_warehouse.loc[(df_warehouse.week == target_week)]\n\n        y_true_sampling = []\n        j = 0\n\n        qty = df.loc[:,'estoque'].as_matrix()\n        my_new_list = [i * 1 for i in qty]\n\n        if(len(my_new_list) < 1):\n            my_new_list = [0] * 2\n        else:\n            my_new_list = my_new_list\n\n        d = np.array(my_new_list) \n        cols = len(d) \n        d = np.append(d,[np.nan]*extra_periods) \n\n        a,p,f = np.full((3,cols+extra_periods),np.nan)\n        q = 1 \n\n        first_occurence = np.argmax(d[:cols]>0)\n        a[0] = d[first_occurence]\n        p[0] = 1 + first_occurence\n        f[0] = a[0]/p[0]\n\n        for t in range(0,cols):        \n            if d[t] > 0:\n                a[t+1] = alpha*d[t] + (1-alpha)*a[t] \n                p[t+1] = alpha*q + (1-alpha)*p[t]\n                f[t+1] = a[t+1]/p[t+1]\n                q = 1           \n            else:\n                a[t+1] = a[t]\n                p[t+1] = p[t]\n                f[t+1] = f[t]\n                q += 1\n\n        a[cols+1:cols+extra_periods] = a[cols]\n        p[cols+1:cols+extra_periods] = p[cols]\n        f[cols+1:cols+extra_periods] = f[cols]\n\n        y_pred_list = list(f)\n        y_pred = y_pred_list[-1]\n\n        y_true = list(df_true.estoque.values)\n\n        if(1 > len(y_true)):\n            y_true_sampling = y_true\n        for j in range(0, (1-len(y_true))):\n            y_true_sampling.append(0)\n        else:\n            for j in range(0, 1):\n                y_true_sampling.append(y_true[j])\n\n        y_true_final += y_true_sampling\n        y_pred_final.append(round(y_pred_list[-1]))\n\n        start_week += 1\n        end_week += 1\n        target_week += 1\n        \n    start_week = 132\n    end_week = 133\n    target_week = 134\n    print(z, i)\n    z += 1\n    \nx_pred_arima = y_pred_final_arima\npred_without_nan_arima = [0 if math.isnan(x) else x for x in x_pred_arima]")


# In[111]:


#count_data
mae = mean_absolute_error(y_true_final_arima, pred_without_nan_arima)
mse = mean_squared_error(y_true_final_arima, pred_without_nan_arima)
rms = sqrt(mse)

print("MAE : ", mae)
print("MSE : ", mse)
print("RMSE : ", rms)


# # fin
