#!/usr/bin/env python
# coding: utf-8

# ## Mordred and Morgan 

# In[2]:


# Computing mordred and morgan features and building a model from these features


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')

import Utils as model_helpers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Computing Features

# In[4]:


half_enantiomer_data = pd.read_csv("half_enantiomer_data.csv")


# In[5]:


# Remove line separaters
half_enantiomer_data["SMILES String"] = half_enantiomer_data["SMILES String"].apply(lambda x : x.replace("\\n", "") and x.replace("\\r", ""))
half_enantiomer_data["SMILES String"] = half_enantiomer_data["SMILES String"].apply(lambda x : x.replace("\\n", ""))


# In[6]:


# Calculate the mordred features
mordred_data = model_helpers.calculate_features(half_enantiomer_data, "mordred")


# In[7]:


# Calculate the morgan features
morgan_data = model_helpers.calculate_features(half_enantiomer_data, "morgan")


# In[8]:


#zero_var_cols = [mordred_data[col] for col in mordred_data.iloc[:,11:] if (mordred_data[col].var() > 0) == True]
#mordred_data.drop(columns)


# In[9]:


# Dataframe with molecules that have mordred and morgan features computed
common_index = mordred_data.index.intersection(morgan_data.index)
mordred_data = mordred_data.loc[common_index]
morgan_data = morgan_data.loc[common_index]


# In[10]:


# Reset index
mordred_data.set_index('Molecule Name').head().iloc[:, 10:];
morgan_data.set_index('Molecule Name').head().iloc[:, 10:];


# In[11]:


# Data frame that has both the mordred and morgan features
both = mordred_data.join(morgan_data.iloc[:,10:], how="inner", rsuffix='morg_')
both.head()


# In[12]:


#Need to drop var columns
print(both.var().max())
print(both.var().min())


# In[13]:


# Gets all Mordred or Mogan features that have numeric values and not Null values
# Joins the final mordred and morgan features 
finite_mordred = model_helpers.finite_features(mordred_data)
finite_morgan = model_helpers.finite_features(morgan_data)
both_features = finite_mordred | finite_morgan


# Model

# In[21]:


# Illustrate the magnitude differences across enantiomeric pairs in the dataset
model_helpers.fold_difference_of_enantiomers(half_enantiomer_data)


# In[15]:


# Gets the appropriate parameter values for mordred model
# Gets the valid features (not null values) from feature data frame and the log_abs values from the feature dataframe
X_morded = mordred_data[finite_mordred]
y = mordred_data['log_abs']
X_morded = X_morded[y < 10]
y_mordred = y[y < 10]
Xn_mordred = pd.DataFrame(StandardScaler().fit_transform(X_morded), index=X_morded.index, columns=X_morded.columns)


# In[16]:


# Gets the appropriate parameter values for Morgan model
# Gets the valid features (not null values) from feature data frame and the log_abs values from the feature dataframe
x_morgan = morgan_data[finite_morgan]
y_morgan = morgan_data["log_abs"]
x_morgan = x_morgan[y_morgan < 10]
y_morgan = y_morgan[y_morgan < 10]
Xn_morgan = pd.DataFrame(StandardScaler().fit_transform(x_morgan), index=x_morgan.index, columns=x_morgan.columns)


# In[25]:


# Model for Morgan data
model_helpers.create_model(Xn_morgan, y_morgan)


# In[26]:


model_helpers.cross_val(Xn_morgan, y_morgan)


# In[24]:


# Model for Mordred data
model_helpers.create_model(Xn_mordred, y_mordred)


# In[23]:


model_helpers.cross_val(Xn_mordred, y_mordred)

