#!/usr/bin/env python
# coding: utf-8

# ## Load Data

# In[31]:


#Gets the updates from the development files that are imported
get_ipython().run_line_magic('load_ext', 'autoreload')


# In[15]:


#Imports the pands library, the math library, and the init class python file
#Last line: updates matlab library that is used in init file
import Utils as model_helpers
import pandas as pd


# In[16]:


# Loads in the data
load_smiles_strings = model_helpers.load_other_smiles(coleman=True)
enantiomer_data = model_helpers.load_data("coleman")


# In[17]:


# Groups each pair of enantiomers by making them have the same number associated with column "N"
# Takes each pair of enantiomers and computes the ratio of the Normalized Detection Thresholds between the two
# Sets the Normalized Detection Thresholds to be of the same type to avoid type errors later on
enantiomer_data['N'] = np.arange(0, enantiomer_data.shape[0]/2, 0.5).astype(int)
enantiomer_data['Normalized Detection Threshold'] = enantiomer_data['Normalized Detection Threshold'].astype('float')
enantiomer_data.head()


# In[18]:


# Adding all new smiles strings from "Other SMILES" column to official "SMILES String" column
enantiomer_data["SMILES String"] = enantiomer_data["Other SMILES"].combine_first(enantiomer_data["SMILES String"]) 


# In[19]:


# Searching for a specific molecule in the dataframe. Checking that SMILES Strings differ
enantiomer_data[enantiomer_data['Molecule Name'].str.contains('lina')]


# In[20]:


enantiomer_data.to_csv("enantiomer_data.csv")


# In[21]:


# Take the absolute log values and harmonic values for each enantiomeric pair
half_log_abs = enantiomer_data.groupby('N').apply(model_helpers.log_abs)
half_det = enantiomer_data.groupby('N').apply(model_helpers.harmonic)


# In[22]:


# Creates a new data frame with just one odorant of each enantiomeric pair from the original dataset 
# adds the absolute value and detection threshold value for remaining odorants from enantiomering pair
half_enantiomer_data = enantiomer_data.iloc[::2].copy()
half_enantiomer_data.loc[:, 'log_abs'] = half_log_abs.values
half_enantiomer_data.loc[:, 'det'] = half_det.values


# In[23]:


# This line makes sure that the rest of the exsisting null values are equal in the new data frame and in the new data frame's 'log_abs' column
assert half_log_abs.isnull().sum() == half_enantiomer_data['log_abs'].isnull().sum()


# In[24]:


# This line checks that log_abs and det columns were added properly
half_enantiomer_data.head()


# In[25]:


# Gets rid of all the invalid SMILES Strings, specifically the duplicates because we don't want to count their perceptual features twice and the "nan" values 
half_enantiomer_data = half_enantiomer_data.drop_duplicates(subset=['SMILES String'])
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['SMILES String'].str.contains('NaN', na=True)]
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['SMILES String'].str.contains('nan', na=True)]


# In[26]:


# Assert statement to ensure that we only have unqiue smiles strings 
assert half_enantiomer_data['SMILES String'].shape == half_enantiomer_data['SMILES String'].unique().shape, "Number of SMILES strings should equal number of unique SMILES strings at this stage"


# In[27]:


# Assert that there are no more nan values in the smiles string column
assert sum(half_enantiomer_data['SMILES String']=='nan') == 0, "There should be no NaN SMILES strings at this point"


# In[28]:


# Gets rid of the rows with a null log_abs value
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['log_abs'].isnull()]


# In[29]:


# Assert that there are no more log_abs of det values with the value null
assert not sum(half_enantiomer_data['log_abs'].isnull())
assert not sum(half_enantiomer_data['det'].isnull())

