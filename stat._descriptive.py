#!/usr/bin/env python
# coding: utf-8

# In[61]:


import random as rd
import math as mt
import matplotlib.pyplot as plt
tab=[]
taille=int(input("entrer le nombre d'element du tableau : "))
for i in range(taille):
    x=rd.randint(0,20)
    tab.append(x)
print("mon tableau : ",tab)
t1=sorted(tab)
print("tableau trié :",t1)
# la moyenne

moyenne=sum(tab)/len(tab)

print("la moyenne est : ",moyenne)

#  les extremes 

print("les valeurs extremes sont ", t1[0]," et ",t1[-1])  

# etendue

e= t1[-1]-t1[0]
print("l'etendue est : ",e)

# la variance
svar=0
for i in range(taille):
    svar+=(tab[i]-moyenne)**2
var=svar/(len(tab)-1)
print("la variance est : ",var)
# l'ecart type 

print("l'ecart type est ", mt.sqrt(var))
plt.hist(t1)


# In[ ]:





# In[ ]:





# In[ ]:




