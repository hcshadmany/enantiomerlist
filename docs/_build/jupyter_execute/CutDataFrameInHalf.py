#!/usr/bin/env python
# coding: utf-8

# ### Cutting Data Frame in Half

# In[ ]:


# Cutting the dataframe in half to use only one molecule per each enantiomeric pair


# In[1]:


#Gets the updates from the development files that are imported
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[13]:


#Imports the pands library, the math library, and the init class python file
#Last line: updates matlab library that is used in init file
import Utils as model_helpers
import numpy as np
import pandas as pd


# In[14]:


enantiomer_data = pd.read_csv("enantiomer_data.csv")


# In[15]:


# Take the absolute log values and harmonic values for each enantiomeric pair
half_log_abs = enantiomer_data.groupby('N').apply(model_helpers.log_abs)
half_det = enantiomer_data.groupby('N').apply(model_helpers.harmonic)


# In[16]:


# Creates a new data frame with just one odorant of each enantiomeric pair from the original dataset 
# adds the absolute value and detection threshold value for remaining odorants from enantiomering pair
half_enantiomer_data = enantiomer_data.iloc[::2].copy()
half_enantiomer_data.loc[:, 'log_abs'] = half_log_abs.values
half_enantiomer_data.loc[:, 'det'] = half_det.values


# In[17]:


# This line makes sure that the rest of the exsisting null values are equal in the new data frame and in the new data frame's 'log_abs' column
assert half_log_abs.isnull().sum() == half_enantiomer_data['log_abs'].isnull().sum()


# In[18]:


# This line checks that log_abs and det columns were added properly
half_enantiomer_data.head()


# In[19]:


# Gets rid of all the invalid SMILES Strings, specifically the duplicates because we don't want to count their perceptual features twice and the "nan" values 
half_enantiomer_data = half_enantiomer_data.drop_duplicates(subset=['SMILES String'])
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['SMILES String'].str.contains('NaN', na=True)]
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['SMILES String'].str.contains('nan', na=True)]


# In[20]:


# Assert statement to ensure that we only have unqiue smiles strings 
assert half_enantiomer_data['SMILES String'].shape == half_enantiomer_data['SMILES String'].unique().shape, "Number of SMILES strings should equal number of unique SMILES strings at this stage"


# In[21]:


# Assert that there are no more nan values in the smiles string column
assert sum(half_enantiomer_data['SMILES String']=='nan') == 0, "There should be no NaN SMILES strings at this point"


# In[22]:


# Gets rid of the rows with a null log_abs value
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['log_abs'].isnull()]


# In[23]:


# Assert that there are no more log_abs of det values with the value null
assert not sum(half_enantiomer_data['log_abs'].isnull())
assert not sum(half_enantiomer_data['det'].isnull())


# In[24]:


half_enantiomer_data.to_csv("half_enantiomer_data.csv")

