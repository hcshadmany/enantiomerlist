#!/usr/bin/env python
# coding: utf-8

# ## 3rd Party Embeddings

# In[ ]:


# Using 3rd party embeddings as features to the model


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')

import Utils as model_helpers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Computing Features

# In[3]:


half_enantiomer_data = pd.read_csv("half_enantiomer_data.csv")


# In[20]:


# Loads in embeddings from 3rd party model to use as features
gme = np.load('../data/thirdparty/enantiomer-embeddings-for-rick.npz', allow_pickle=True) # Load the file
gme = gme['embeddings'].item() # Extract the data
gme = {k: v.squeeze() for k, v in gme.items()} # Flatten the arrays
gme_df = pd.DataFrame(gme).T  # Turn into a dataframe


# In[5]:


gme_df.head()


# In[6]:


# Make copy of original data and set the index to match that of the gme model
half_enantiomer_data_copy = half_enantiomer_data
half_enantiomer_data_copy = half_enantiomer_data_copy.set_index("SMILES String")


# In[7]:


# Keep the columns in the gme df that match the index names in original dataset
common_index = half_enantiomer_data_copy.index.intersection(gme_df.index)
half_enantiomer_data_copy = half_enantiomer_data_copy.loc[common_index]
gme_df = gme_df.loc[common_index]


# In[8]:


# Combine original dataset with gme df
g_model_embeddings = half_enantiomer_data_copy.join(gme_df, how="inner")


# In[9]:


# Reset the index to be "Moecule Name"
g_model_embeddings = g_model_embeddings.set_index("Molecule Name")


# In[10]:


g_model_embeddings.head()


# In[30]:


assert ((g_model_embeddings.iloc[:,15:].var() <= 0).sum() == 0), "This should be 0 if not, get rid of columns with 0 varience"


# Model

# In[15]:


# Illustrate the magnitude differences across enantiomeric pairs in the dataset
model_helpers.fold_difference_of_enantiomers(half_enantiomer_data)


# In[28]:


x_gme = g_model_embeddings.iloc[:,15:]
y_gme = g_model_embeddings["log_abs"]
Xn_gme = pd.DataFrame(StandardScaler().fit_transform(x_gme), index=x_gme.index, columns=x_gme.columns)


# In[29]:


model_helpers.create_model(Xn_gme, y_gme)


# In[ ]:




