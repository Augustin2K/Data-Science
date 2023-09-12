#!/usr/bin/env python
# coding: utf-8

# # dataset pour l'analyse economique d'une banque

# In[207]:


# importation des modumes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[208]:


# importation du fichier csv

chemin="C:/Users/ASUS/Documents/formation_data_science/bank+marketing/bank/bank-full.csv"
bank_df= pd.read_csv(chemin,sep=";")


# In[209]:


# description rapide de la dataset

descript=bank_df.describe()
descript


# In[210]:


bank_df.head(10)


# In[211]:


bank_df.shape


# In[212]:


bank_df['marital'].value_counts().plot.bar()


# In[213]:


bank_df['education'].value_counts().plot.bar()


# In[214]:


bank_df['job'].value_counts().plot.bar()


# In[215]:


bank_df['housing'].value_counts().plot.bar()


# In[216]:


sns.histplot(data=bank_df,x="housing")


# In[217]:


bank_df['marital'].value_counts().plot.bar()


# In[242]:


bank_df.groupby(["marital","y"]).max()


# In[219]:


sns.pairplot(bank_df,hue="poutcome")


# In[220]:


sns.boxplot(data=bank_df,x='y',y='duration')


# In[221]:


sns.histplot(data=bank_df,x="poutcome")


# In[222]:


sns.pairplot(bank_df,hue="y")


# In[223]:


sns.pairplot(bank_df,hue="education")


# # le resultat souhaiter est donné par la variable "Y " qui determine si le client a souscrit un depot à terme. c'est une variable binaire("yes" or "no")

# In[224]:


sns.histplot(data=bank_df,x="y")


# # le nbre des gens dont la duree d'appel est inferieur à 30sec

# In[225]:



Ont_ignorE=bank_df[bank_df['duration']<30]
Ont_ignorE.shape


# In[226]:


a=Ont_ignorE['y']


# In[227]:


i=0
for j in a:
    if j==1:
        i+=1
print((1-i/len(a))*100,"% des gens dont l'appal a duree moins de 30sec et qui ont souscrit à l'offre")


# # preparation de la dataset pour l'apprentissage

# In[228]:



bank_df.head(10)


# In[229]:


bank_df.columns


# In[230]:


bank_df=bank_df.drop(["month"],axis=1)


# In[231]:


bank_df["marital_encode"]=pd.factorize(bank_df["marital"])[0]
bank_df["job_encode"]=pd.factorize(bank_df["job"])[0]
bank_df["education_encode"]=pd.factorize(bank_df["education"])[0]
bank_df["default_encode"]=pd.factorize(bank_df["default"])[0]
bank_df["housing_encode"]=pd.factorize(bank_df["housing"])[0]
bank_df["loan_encode"]=pd.factorize(bank_df["loan"])[0]
bank_df["contact_encode"]=pd.factorize(bank_df["contact"])[0]
bank_df["poutcome_encode"]=pd.factorize(bank_df["poutcome"])[0]
bank_df["y_encode"]=pd.factorize(bank_df["y"])[0]


# In[232]:


bank_df.head(10)


# In[233]:


bank_df=bank_df.drop(["job","marital","education","default","housing","loan","contact","poutcome","y"],axis=1)


# In[234]:


bank_df.head(10)


# In[235]:


bank_df.describe()


# In[236]:


from sklearn import neighbors
from sklearn.model_selection import train_test_split


# In[237]:


sns.pairplot(bank_df,hue="y_encode")


# In[238]:


bank_df.head(5)


# In[239]:


X=bank_df.drop(["y_encode"],axis=1)
Y=bank_df['y_encode']
sns.barplot(data=bank_df, x="housing_encode", y="marital_encode", hue="y_encode")


# In[240]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=0)


# In[241]:


tab_voisin=[]
tab_fiabilite=[]
for i in range(1,50):
    classe=neighbors.KNeighborsClassifier(i)
    classe.fit(X_train,Y_train.values.ravel())
    table_predict=classe.predict(X_test)
   
    tab=[]
    compt=0
    Y=Y_test.values.ravel()
    for j in table_predict:
    
        if table_predict[j]==Y[j]:
            compt+=1
    tab_fiabilite.append(compt/len(table_predict))
    tab_voisin.append(i)
plt.plot(tab_voisin,tab_fiabilite)


# # pour la courbe ci-haute, on peut dire, pour que le systeme soit plus performant dans la prediction, il faut que le nombre de voisin "K" soit compris entre 5 et 12

# 
# # ________________________________________________________________________

# # Regression logistique

# In[ ]:




