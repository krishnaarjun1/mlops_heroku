#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# In[26]:


data=pd.read_csv('suv_data.csv')


# In[27]:


data


# In[28]:


# getting the null value information


# In[29]:


data.isnull().sum()


# In[30]:


# null value imputation


# In[31]:


data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])
data['Age']=data['Age'].fillna(data['Age'].mean())
data['EstimatedSalary']=data['EstimatedSalary'].fillna(data['EstimatedSalary'].mean())


# In[32]:


data.isnull().sum()


# In[33]:


data


# In[34]:


#feature extraction 

X=data.iloc[:,0:3]


# In[35]:


X


# In[36]:


# converting the gender into numeirc

from sklearn.preprocessing import LabelEncoder


# In[37]:


le=LabelEncoder()


# In[38]:


X['Gender']=le.fit_transform(X['Gender'])


# In[39]:


X


# In[40]:


y=data.iloc[:,-1]


# In[41]:


y


# In[42]:


from sklearn.linear_model import LogisticRegression


# In[44]:


model=LogisticRegression()


# In[45]:


model=model.fit(X,y)


# In[49]:


# saving the model into the disk

pickle.dump(model, open('model.pkl','wb'))


# In[50]:


# loading the model

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[0,19,12000]]))


# In[ ]:




