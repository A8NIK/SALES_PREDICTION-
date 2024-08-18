#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


data= pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (17)\Advertising.csv")
data


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (17)\Advertising.csv")
feature_cols = ['TV', 'Radio', 'Newspaper']
models = {}

for feature in feature_cols:
    x = data[[feature]]
    y = data['Sales']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    models[feature] = model 
    y_prediction = models[feature].predict(x_test)
    
    mae = mean_absolute_error(y_test, y_prediction)
    mse = mean_squared_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)

    print(f"Feature: {feature}")
    print(f'MAE:{mae:.2f}, MSE:{mse:.2f}, R-SQUARE:{r2:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

feature_idx = 0
for feature in feature_cols:
    x = data[[feature]]
    y = data['Sales']
    
    axes[feature_idx].scatter(x, y, color='blue')  # Or any desired default color

    axes[feature_idx].set_xlabel(feature, fontsize=10)
    axes[feature_idx].set_ylabel('Sales', fontsize=10)
    axes[feature_idx].set_title(f"Sales vs. {feature}", fontsize=12, fontweight='bold')

    feature_idx += 1

plt.suptitle('Sales Prediction Using Linear Regression (TV, Radio, Newspaper)',fontsize=30, fontweight='bold', color='BLACK', fontname='Georgia')  # Added suptitle
plt.tight_layout()

plt.show()


# In[54]:


import seaborn as sns 
data = pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (17)\Advertising.csv")
corr_matrix=data.corr()
sns.heatmap(corr_matrix,cmap='plasma',linewidth=0.005)
plt.show


# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet

data = pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (17)\Advertising.csv")
feature_cols = ['TV', 'Radio', 'Newspaper']
models = {}

for feature in feature_cols:
    x = data[[feature]]
    y = data['Sales']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = ElasticNet(alpha=5.0, l1_ratio=1)
    model.fit(x_train, y_train)
    models[feature] = model 
    y_prediction = models[feature].predict(x_test)
    
    mae = mean_absolute_error(y_test, y_prediction)
    mse = mean_squared_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)

    print(f"Feature: {feature}")
    print(f'MAE:{mae:.2f}, MSE:{mse:.2f}, R-SQUARE:{r2:.4f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

feature_idx = 0
for feature in feature_cols:
    x = data[[feature]]
    y = data['Sales']
    
    axes[feature_idx].scatter(x, y, color='blue')  # Or any desired default color

    axes[feature_idx].set_xlabel(feature, fontsize=10)
    axes[feature_idx].set_ylabel('Sales', fontsize=10)
    axes[feature_idx].set_title(f"Sales vs. {feature}", fontsize=12, fontweight='bold')

    feature_idx += 1

plt.suptitle('Sales Prediction Using Linear Regression (TV, Radio, Newspaper)',fontsize=30, fontweight='bold', color='BLACK', fontname='Georgia')  # Added suptitle
plt.tight_layout()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




