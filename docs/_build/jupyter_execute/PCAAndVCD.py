#!/usr/bin/env python
# coding: utf-8

# ## PCA and VCD (Could delete since there were no matches)

# In[ ]:


# Comput PCS and VCD features to feed into model


# In[2]:


import pandas as pd


# In[3]:


half_enantiomer_data = pd.read_csv("half_enantiomer_data.csv")


# In[6]:


# Read in pca and vcd dataframe to use as features
pca_vcd_values = pd.read_csv("../data/vcd/pca_vcd_values.csv")


# In[10]:


pca_vcd_values


# In[11]:


# Set index to Molecule Name
pca_vcd_values = pca_vcd_values.set_index('Molecule Name')


# In[12]:


pca_vcd_values.head()


# In[16]:


# Copy original dataframe
half_enantiomer_data_copy = half_enantiomer_data
half_enantiomer_data_copy;


# In[20]:


# Keep the columns in the gme df that match the index names in original dataset
common_index = half_enantiomer_data_copy.index.intersection(pca_vcd_values.index)
half_enantiomer_data_copy = half_enantiomer_data_copy.loc[common_index]
pca_vcd_features = pca_vcd_values.loc[common_index]


# In[18]:


# Combine original dataset with pca_vcd
pca_vcd = half_enantiomer_data_copy.join(pca_vcd_features, how="inner")


# In[19]:


pca_vcd

