#!/usr/bin/env python
# coding: utf-8

# **N/B PCA is not a part of machine learning**

# Wine dataset has many wine feature each row consist to a wine and columns are wine charateristics UCIML dataset.
# Say the wine shop owner(Wine merchant) asked you to first do some preliminary work of clustering to identify diverse segments
# of customers grouped by similarity due to wines they prefer. each of the segment corresponse to same group of 
# customers with same preferences.However in this notebook we are interested in dimensionality reduction say the second mission
# i.e ending with a smaller amount of features and build a predictive model on these new features to predict the segment which
# each wine belongs to , this can help recommend wines to new clients hence help optimize the sale/profit the business.

# ### Importing libraries

# In[13]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


# #### Importing dataset

# In[14]:


dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values  #retain only array values no colnames
Y = dataset.iloc[:,-1].values


# In[15]:


print(X)


# ### Splitting datset

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# ### Feature Scaling

# In[24]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#apply only transform method apply on test set to avoid information leakage on test set
X_test = sc.transform(X_test)  


# In[25]:


print(X_train)


# ### Applying PCA

# In[27]:


from sklearn.decomposition import PCA
#creating instance of the class PCA
#you can vary the components from 2 if performances are not good
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
#apply only transform method on test set to avoid information leakage on test set
X_test = pca.transform(X_test)


# In[28]:


print(X_train)


# ### Training the logistic regression on the training set 

# In[29]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# ### Making confusion matrix

# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# * PCA not only reduce dimensionality but also improves your model greatly 

# ### Visualizing the training set results 

# In[31]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# ### Visualizing the test set results 

# In[32]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[ ]:




