#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# In[2]:


x,y =make_regression(n_samples=100, n_features=1,noise=10)
plt.scatter(x,y)


# In[3]:


x.shape


# In[4]:


y.shape

y=y.reshape(y.shape[0],1)
y.shape
# In[121]:


X=np.hstack((x,np.ones(x.shape)))


# In[122]:


theta=np.random.randn(2,1)
theta


# In[123]:


def model(X, theta):
    return X.dot(theta)


# In[124]:


plt.plot(x,model(X,theta),c='r')
plt.scatter(x,y)


# In[125]:


def fcout(X,y,theta):
    m= len(y)
    return 1/(2*m)*np.sum((model(X,theta)-y)**2,)


# In[126]:


fcout(X,y,theta)


# # descente de gradient

# In[128]:


def grad(X,y,theta):
    m=len(y)
    return 1/(2*m)* X.T.dot(model(X,theta)-y)


# In[129]:


grad(X,y,theta)


# In[130]:


def descGrad(X,y,theta,a,iteration):
    hist=np.zeros(iteration)
    for i in range(0,iteration):
        theta=theta-a*grad(X,y,theta)
        hist[i]=fcout(X,y,theta)
    return theta,hist
        


# In[131]:


theta,hist=descGrad(X,y,theta,0.2,1000)
theta


# In[132]:


pred=model(X,theta)
plt.plot(x,pred,c='r')
plt.scatter(x,y)


# In[133]:


plt.plot(range(1000),hist)


# In[134]:


def coef_performance(y,Pred):
    u=((y-Pred)**2).sum()
    v= ((y-y.mean())**2).sum()
    return 1- u/v


# In[135]:


print(coef_performance(y,pred)*100," %")


# In[ ]:




