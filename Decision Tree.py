#!/usr/bin/env python
# coding: utf-8

# ## Objective

# The objective of this assignment is to apply Decision Tree Classification to a given dataset, analyse the performance of the model, and interpret the results.

# ## Tasks

# 1. Data Preparation:
# Load the dataset into your preferred data analysis environment (e.g., Python with libraries like Pandas and NumPy).
# 2. Exploratory Data Analysis (EDA):
# Perform exploratory data analysis to understand the structure of the dataset.
# Check for missing values, outliers, and inconsistencies in the data.
# Visualize the distribution of features, including histograms, box plots, and correlation matrices.
# 3. Feature Engineering:
# If necessary, perform feature engineering techniques such as encoding categorical variables, scaling numerical features, or handling missing values.
# 4. Decision Tree Classification:
# Split the dataset into training and testing sets (e.g., using an 80-20 split).
# Implement a Decision Tree Classification model using a library like scikit-learn.
# Train the model on the training set and evaluate its performance on the testing set using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score, ROC-AUC).
# 5. Hyperparameter Tuning:
# Perform hyperparameter tuning to optimize the Decision Tree model. Experiment with different hyperparameters such as maximum depth, minimum samples split, and criterion.
# 6. Model Evaluation and Analysis:
# Analyse the performance of the Decision Tree model using the evaluation metrics obtained.
# Visualize the decision tree structure to understand the rules learned by the model and identify important features
# 

# ## Answer

# In[42]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder


# Data preparation

# In[2]:


# Read the .xlsx file as a Pandas dataframe
file = pd.ExcelFile('heart_disease.xlsx')
abr = pd.read_excel(file, 'Description')          # Coloumn headers abbreviation sheet
df = pd.read_excel(file, 'Heart_disease')         # Main dataframe sheet


# In[3]:


print(abr)


# In[4]:


df


#  EDA & feature engineering

# In[5]:


df.describe()       # Statistical values 


# In[6]:


# Check for null values
df.isnull().any()


# In[7]:


df.dropna(inplace=True)
df


# In[8]:


# Check for duplicates
df.duplicated().any()


# In[9]:


df.drop_duplicates(inplace=True)
df


# In[10]:


# Labelling data
df['cp'].unique()


# In[12]:


# Label encoding on Chest pain type data
lab_enc= LabelEncoder()
df['cp']= lab_enc.fit_transform(df[['cp']])
df


# In[13]:


df['restecg'].unique()


# In[14]:


# Label encoding ecg observation
df['restecg']= lab_enc.fit_transform(df[['restecg']])
df


# In[15]:


# Label encoding for slope and thal
df['slope']= lab_enc.fit_transform(df[['slope']])
df['thal']= lab_enc.fit_transform(df[['thal']])
df


# In[22]:


df['exang'].unique()


# In[25]:


df['sex']= lab_enc.fit_transform(df[['sex']])
df['fbs']= lab_enc.fit_transform(df[['fbs']])


# In[28]:


# For exang values
mapping = {'False':False, 'True':True, 'FALSE':False}
df['exang'] = df['exang'].map(mapping).fillna(df['exang'])
df['exang'].unique()


# In[29]:


df['exang']= lab_enc.fit_transform(df[['exang']])
df


# In[30]:


# Setting features and target
target = df[['num']]
features = df.drop('num', axis=1)
features


# In[34]:


# Detection of outliers using visualization of distribution of features

plt.figure(figsize=(12,6))
sns.histplot(features, kde=True)
plt.xlim([-20, 400])
plt.show()


# In[35]:


# box plot

plt.figure(figsize=(12,6))
sns.boxplot(data=features)
plt.title('Box plot of features')
plt.show()


# In[36]:


# Correlation matrices

corr = features.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()


# In[37]:


# Splitting into training and testing data
x_train,x_test,y_train,y_test= train_test_split(features,target,train_size=0.75,random_state=100)


# Decision Tree Classification

# In[38]:


# Initiating decision tree model
dec_tree= DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3)


# In[39]:


dec_tree.fit(x_train,y_train)


# In[40]:


y_pred=dec_tree.predict(x_test)


# In[41]:


# Checking accuracy score
accuracy_score(y_test,y_pred)


# In[50]:


# Classification report
print(classification_report(y_test, y_pred))


# In[55]:


# ROC AUC score

from sklearn.metrics import roc_auc_score

y_pred_proba = dec_tree.predict_proba(x_test)
y_pred_proba


# In[56]:


roc_auc_score(y_test,y_pred_proba,multi_class="ovr")


# Visualizing decision tree structure

# In[44]:


plt.figure(figsize=(15,15))
plot_tree(dec_tree,filled=True,rounded=True);


# In[45]:


from sklearn import tree
print(tree.export_text(dec_tree))


# Hyperparameter tuning

# In[46]:


# Hyperparameter tuning

params= {'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth':[1,2,3,4,5]}

grid_search= GridSearchCV(dec_tree,params,verbose=2)


# In[47]:


grid_search.fit(x_train,y_train)


# In[48]:


grid_search.best_params_


# In[ ]:




