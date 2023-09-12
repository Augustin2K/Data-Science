#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import matplotlib.pyplot as plt


# In[71]:


data = pd.read_csv("C:/Users/A2K Scorpion/Documents/Formation/maternal_health_risk/Maternal Health Risk Data Set.csv")


# In[72]:


data.head()


# In[73]:


#Age: age
#SystolicBP: pression arterielle Systolic
#DistolicBP: pression diastolic
#BP: Glycemie
#BodyTemp: Temperature corporelle
#HeartRate: Frequence cardiaque
#RiskLevel: Niveau de risque


# In[74]:


data.describe()


# In[75]:


import seaborn as sns


# In[76]:


sns.pairplot(data, hue="RiskLevel")


# In[77]:


sns.histplot(data, x="RiskLevel")


# # Preparation

# In[78]:


data["RiskLevel_encode"] = pd.factorize(data["RiskLevel"])[0]


# In[79]:


data.head()


# In[80]:


train_df = data[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel_encode"]]


# In[81]:


train_df.head()


# In[82]:


from sklearn import neighbors as ng 


# In[83]:


X = train_df.iloc[:,0:6]
Y = train_df.iloc[:,[6]]


# In[84]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)


# In[101]:


knn = ng.KNeighborsClassifier(4)
knn.fit(X_train, Y_train.values.ravel())


# In[102]:


knn.predict(X_test)


# In[103]:


prd = knn.predict(X_test)
n = 0
Y_test_arr = Y_test.values.ravel()
for i in prd:
    if prd[i] == Y_test_arr[i]:
        n += 1 
n/len(prd)


# In[104]:


tab1 = []
tab2 = []
for i in range(1, 100):
    knn = ng.KNeighborsClassifier(i)
    tab1.append(i)
    knn.fit(X_train, Y_train.values.ravel())
    knn.predict(X_test)
    prd = knn.predict(X_test)
    n = 0
    Y_test_arr = Y_test.values.ravel()
    for i in prd:
        if prd[i] == Y_test_arr[i]:
            n += 1 
    tab2.append(n/len(prd))
    


# # Courbe d'apprentissage

# In[105]:


plt.plot(tab1,tab2)


# In[ ]:




