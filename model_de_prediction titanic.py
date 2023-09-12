#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


chemin= "C:/Users/ASUS/Documents/watTitanic.xlsx" 
donnee=pd.read_excel(chemin)
donnee


# In[6]:


donnee=donnee.drop(["PassengerId" ,"Name" ,"SibSp" ,"Parch" ,"Ticket" ,"Fare" ,"Cabin" ,"Embarked"],axis=1)
donnee


# In[7]:


donnee["Sex"].replace(["male","female"],[0,1], inplace=True)
donnee


# In[8]:


donnee.describe()


# In[9]:


donnee=donnee.dropna()
model=KNeighborsClassifier()

y=donnee["Survived"]
X=donnee.drop("Survived",axis=1)
donnee


# In[10]:


model.fit(X,y)
model.score(X,y)


# In[11]:


def survie(model,pclass,sex,age):
   
    df=np.array([a,b,c]).reshape(1,3)
    pred=model.predict(df)
    proba=model.predict_proba(df)
    return pred


# In[12]:


a=int(input("class de voyage : "))
b=int(input("sexe : "))
while(b!=0 and b!=1):
    print("sexe non reconnu. tapez 0 si vous etes un homme et 1 sinon")
    b=int(input("sexe : "))
c=int(input("votre age : "))
survie(model,a,b,c)


# In[ ]:




