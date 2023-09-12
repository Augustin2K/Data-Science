#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


data = pd.read_csv("C:/Users/A2K Scorpion/Documents/Formation/drug200.csv")


# In[10]:


data.head()


# In[11]:


data.describe()


# In[12]:


import seaborn as sns


# In[13]:


sns.pairplot(data, hue="Drug")


# In[14]:


sns.pairplot(data, hue="Sex")


# In[15]:


sns.pairplot(data, hue="Cholesterol")


# In[16]:


sns.pairplot(data, hue="BP")


# In[17]:


pd.factorize(data['Sex'])


# In[18]:


pd.factorize(data['BP'])


# In[19]:


data["BP_encode"]=pd.factorize(data['BP'])[0]


# In[20]:


data.head()


# In[21]:


sns.pairplot(data, hue ="BP")


# In[22]:


data["Sex_encode"]=pd.factorize(data['Sex'])[0]
data["Cholesterol_encode"]=pd.factorize(data['Cholesterol'])[0]
data["Drug_encode"]=pd.factorize(data['Drug'])[0]


# In[23]:


data.head()


# # preparation

# In[24]:


train_df = data[["Age","Sex_encode","BP_encode","Cholesterol_encode","Na_to_K","Drug_encode"]]


# In[25]:


train_df.head


# In[40]:


sns.histplot(data, x="Age")


# In[41]:


sns.histplot(data, x="Sex")


# In[42]:


sns.histplot(data, x="Drug")


# In[2]:


import math


# In[ ]:


#Algo_test 
def distance(X1, X2):
    return math.sqrt(())

def compute (xnew, ds:list, k:int):
    dst = []
    for e in dst
        d = distance(xnew, e)
        dst.append(d)
        dst = dst.sort()
    return dst[:k]
        
def vote (dst:list):
    newSet = set(dst)
    for e in newSet
       elt = dst.count(e)
        
#    


# In[4]:


from sklearn import neighbors


# In[46]:


train_df.head()


# In[40]:


from sklearn import neighbors as ng


# In[41]:


X = train_df.iloc[:,0:5]


# In[85]:


Y = train_df.iloc[:,[5]]


# In[86]:


Y.shape


# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 243)
X_train


# In[88]:


clf = ng.KNeighborsClassifier(12)
clf.fit(X_train, Y_train.values.ravel())


# In[89]:


clf.predict(X_test)


# In[90]:



prd = clf.predict(X_test)
n = 0
Y_test_arr = Y_test.values.ravel()
for i in prd:
    if prd[i] == Y_test_arr[i]:
        n += 1       
n/len(prd)


# In[91]:



tab1=[]
tab2=[]
for i in range(1,51):
    clf = ng.KNeighborsClassifier(i)
    tab1.append(i)
    clf.fit(X_train, Y_train.values.ravel())
    clf.predict(X_test)
    prd = clf.predict(X_test)
    n = 0
    Y_test_arr = Y_test.values.ravel()
    for i in prd:
        if prd[i] == Y_test_arr[i]:
            n += 1       
    tab2.append(n/len(prd))

    


# In[79]:


import matplotlib.pyplot as plt
plt.plot(tab1,tab2)


# In[ ]:




