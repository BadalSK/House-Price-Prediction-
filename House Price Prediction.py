#!/usr/bin/env python
# coding: utf-8

# In[44]:


'''Objective:- Machine learning model (XGBoost) to predict house prices using features like:

Number of bedrooms (BedroomAbvGr)

Living area size (GrLivArea)

goal is:

"Given the house features, predict its price as accurately as possible."'''


# Data Cleaning + XGBoost

# In[31]:


get_ipython().system('pip install xgboost')
import pandas as pd
import psycopg2


# In[18]:


# Connecting to PostgreSQL

conn = psycopg2.connect(
      dbname="real_estate",
      user = "postgres",
    password="123456789",
    host="localhost"
);


# In[19]:


#Load the data

data = pd.read_sql("SELECT BedroomAbvGr, GrLivArea, SalePrice, LotFrontage FROM house_prices",conn)


# In[20]:


## data2 =pd.read_csv("C:/Workspace/datasets/train (House Price) CLEAN.csv")


# In[21]:


#Data Analyzing
data.head()


# In[22]:


data.describe()


# In[23]:


data.isnull().sum()


# In[24]:


# Clean the Data

#Handle missing values
data['bedroomabvgr'].fillna(data['bedroomabvgr'].median(),inplace=True)

"""When to Use inplace=True?
✅ Use inplace=True when you want to modify the original DataFrame.
✅ Avoid it if you need a copy of the DataFrame before making changes."""

#Filter Realistic Homes
data = data[(data['bedroomabvgr']>0) &(data['grlivarea']<5000)]


# data.columns

# In[25]:


# OUTLIER Removal (Beyond Your Current Filters)
## Remove homes with extreme outliers (adjust thresholds as needed)
data = data[
    (data['bedroomabvgr']>0) &
    (data['grlivarea']<5000) &
    (data['saleprice']<500000) # Filter ultra-high prices
]


# In[26]:


data.info()


# ## Log Transform Skewed Features (Improves Model Performance)

# In[ ]:





# In[27]:


import numpy as np

# Log-transform skewed features (common for prices/areas)
data['LogSalePrice'] = np.log1p(data['saleprice'])
data['LogGrLivArea'] = np.log1p(data['grlivarea'])
data['Logbedroomabvgr'] = np.log1p(data['bedroomabvgr'])

# Convert back to actual prices [np.expm1(value) is reverse of log1p(value)].


# In[28]:


data['LogSalePrice']


# ## Train XGBoost Model

# ##### XGBoost Model – Simplified Explanation
# XGBoost (eXtreme Gradient Boosting) is a powerful, fast, and efficient machine learning algorithm mainly used for classification and regression problems. It improves gradient boosting by optimizing speed and performance.
# 
# ##### How XGBoost Works (In Simple Terms)
# It builds decision trees sequentially (one after another).
# 
# Each new tree fixes errors made by previous trees.
# 
# It gives higher weight to misclassified points.
# 
# Finally, it combines all trees to make a strong prediction model.
# 
# ##### Key Hyperparameters (Tuning Tips)
# n_estimators: Number of trees (higher = better accuracy, but slower).
# 
# learning_rate: Controls step size (0.1 is a good start).
# 
# max_depth: Limits tree depth (prevents overfitting).
# 
# subsample: Takes a fraction of data per tree (helps generalization).
# 
# ##### Why Use XGBoost?
# ✔ Faster than other algorithms
# ✔ Handles missing values
# ✔ Works well for structured/tabular data
# ✔ Prevents overfitting with regularization
# 
# #### Key Points in Classification
# Instead of XGBRegressor, we use XGBClassifier.
# 
# The target variable (y) must be binary (0 or 1) for classification.
# 
# Evaluation Metric (logloss): Measures classification error.
# 
# Hyperparameter tuning is similar to regression.
# 
# ##### Why Use XGBoost for Classification?
# ✔ Handles large datasets efficiently
# ✔ Works well with missing values
# ✔ Prevents overfitting using regularization
# ✔ Fast and accurate compared to other classifiers

# In[34]:


# Train XGBoost Model For_---Regression 

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

x = data[['Logbedroomabvgr','LogGrLivArea']]
y = data['LogSalePrice']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)

model = XGBRegressor()
model.fit(x_train, y_train)


# In[39]:


# Predict and Evalute
from sklearn.metrics import mean_squared_error

predictions = model.predict(x_test)

#Convert predicted log prices back to original prices
predicted_prices = np.expm1(predictions)

#Also if y_test is in log form
actual_prices = np.expm1(y_test)

rmse = (mean_squared_error(actual_prices,predicted_prices))**0.5

#RMSE (Root Mean Squared Error) tells you how far off your predictions are from the actual prices.

print(f"RMSE: ₹{rmse:,.2f}" )


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot: Actual vs Predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()



# '''✅ What This Shows:
# Each dot = one house
# 
# X-axis = real price
# 
# Y-axis = predicted price
# 
# Red dashed line = perfect prediction'''

# ## Improveing Model

# In[48]:


#  a) Feature Engineering
#Adding more Column
data1 = pd.read_csv("C:/Workspace/datasets/train (House Price).csv")


# In[52]:


print(data1.columns)


# In[ ]:





# In[59]:


data = data1[["BedroomAbvGr", "GrLivArea", 'OverallQual', 'YearBuilt', 'SalePrice']]


# In[ ]:





# In[60]:


data.columns


# In[62]:


# Create New Features
data.loc[:,'Age'] = 2025 - data['YearBuilt'] #How ols is the house?
data.loc[:,'PricePerSqFt'] = data['SalePrice']/data['GrLivArea']


# In[64]:


data.info()


# In[65]:


data.isnull().sum()


# In[68]:


# b. Hyperparameter Tuning
#Optimize XGBoost with GridSearchCV

'''Why use GridSearchCV?
Because machine learning models like XGBoost have hyperparameters 
(like tree depth, number of trees) that greatly affect performance.

You don’t know the best values in advance 
— so you try multiple combinations systematically and 
pick the best one using cross-validation.'''

from sklearn.model_selection import GridSearchCV  

param_grid = {  
    'n_estimators': [50, 100],  # Test 2 values (Try 50 and 100 trees)  
    'max_depth': [3, 6]  # Try tree depths of 3 and 6
}  
# Do 3-fold cross-validation for each combination
model = GridSearchCV(XGBRegressor(), param_grid, cv=3)  
model.fit(x_train, y_train)  
print(f"Best Params: {model.best_params_}")  

'''This will:

-Train 4 models (2x2 combos),

-Validate them using 3-fold CV,

-Pick the best one based on validation performance.

'''


# In[69]:


# c. Model Interpretation

from xgboost import plot_importance  
plot_importance(model.best_estimator_)  
plt.show()  


# In[71]:


#2. Visualize Results 
#a. Actual vs. Predicted Plot

plt.scatter(y_test, predictions)  
plt.xlabel("Actual Prices")  
plt.ylabel("Predicted Prices")  
plt.title("House Price Predictions")  
plt.show()  


# In[72]:


# b. Error Analysis
#Find the worst predictions (e.g., houses where the model was off by ₹10L+):
error = np.abs(predictions - y_test)
worst_predictions = x_test[error > 10_00_000] # Customize threshold 


# In[75]:


worst_predictions = x_test[error > 5_00_000]  # Lower threshold  
print(worst_predictions.head())  


# In[77]:





# ## Streamlit

# In[81]:


get_ipython().system('pip install streamlit')

import streamlit as st

st.title("House Price Predictor")  
bedrooms = st.slider("Bedrooms",1,5)
sqft = st.slider("Living Area (sqft)", 500, 5000) 

prediction = model.predict([[bedrooms, sqft]])[0]  
st.write(f"Predicted Price: ₹{prediction:,.2f}")  




# In[ ]:




