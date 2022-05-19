#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from TF_data_prepro import*


# In[2]:


## load data ##
path=r'df_clean.pkl'
df= pd.read_pickle(path)


# In[3]:


## set target ##
pd.set_option('display.max_columns', None)
df['target']=df['price per unit']
df.reset_index(inplace=True,drop=True)
df


# In[4]:


all_features,num,cat=feature_col_clean_split(df)
X=num.drop(columns='price per unit').values
y=num['target'].values


# In[62]:


num


# In[63]:


cat


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[6]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[8]:


X_train.shape


# In[27]:


model=Sequential()

model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))


model.add((Dense(1)))
model.compile(optimizer='adam',loss='mse')


# In[45]:


model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=256,epochs=1200)


# In[46]:


losses=pd.DataFrame(model.history.history)


# In[47]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,30))
losses.plot()


# In[48]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[49]:


predictions=model.predict(X_test)


# In[50]:


np.sqrt(mean_squared_error(y_test,predictions))


# In[51]:


mean_absolute_error(y_test,predictions)


# In[52]:


df['price per unit'].describe()


# In[53]:


explained_variance_score(y_test,predictions)


# In[54]:


plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')


# In[55]:


single_feedthrough=num.drop('price per unit',axis=1).iloc[0]


# In[56]:


single_feedthrough=scaler.transform(single_feedthrough.values.reshape(-1,10))


# In[57]:


model.predict(single_feedthrough)


# In[58]:


df.head(1)


# In[59]:


y_test.shape


# In[60]:


errors = y_test.reshape(157, 1) - predictions


# In[61]:


import seaborn as sns
sns.distplot(errors)


# In[ ]:




