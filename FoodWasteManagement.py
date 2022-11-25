#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv(r"C:\Users\Yash\OneDrive\Documents\FoodWasteManagement.csv")


# In[3]:


dataset


# In[4]:


ds = dataset.columns


# In[5]:


print(ds)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(dataset.loc[:,dataset.columns != 'Result'], dataset['Result'],test_size = 0.30, random_state=1) 


# In[8]:


from sklearn.neighbors import KNeighborsClassifier


# In[9]:


training_accuracy = []
test_accuracy = []


# In[10]:


neighbors_settings = range(1,11)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train, Y_train)
    training_accuracy.append(knn.score(X_train, Y_train))
    test_accuracy.append(knn.score(X_test,Y_test))


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


plt.plot(neighbors_settings, training_accuracy, label = "training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("N-Neighbors")
plt.legend()
plt.savefig('knn_compare_model')


# In[13]:


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, Y_train)
print('\nAccuracy on training set: {:.2f}'.format(knn.score(X_train, Y_train)))
print('\nAccuracy on test set: {:.2f}\n'.format(knn.score(X_test, Y_test)))


# In[14]:


from sklearn.datasets import make_blobs


# In[15]:


X, Y = make_blobs(n_samples = 1000, centers = 2, n_features = 12, random_state = 2)


# In[16]:


knn.fit(X,Y)


# In[25]:


new_input = [[28,28,1130,1600,1808000,28,28,784,364,420,1.15,53.571]]
new_output = knn.predict(new_input)
print("\n'1' indicate you wasted food and '0' indicate you have not wasted food... \n")
print("User Input Datset:- ",new_input,"\n\nPredicted Result is:- ", new_output)


# In[21]:


import pickle


# In[19]:


pickle.dump(knn,open('FoodWasteModel.pkl','wb'))


# In[ ]:





# In[ ]:




