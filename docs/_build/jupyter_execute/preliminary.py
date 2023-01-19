#!/usr/bin/env python
# coding: utf-8

# ## Preliminaries

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[8]:


import __init__ as a


# ## Load data from the DREAM challenge

# In[5]:


# Load the CIDs from DREAM data set
dream_CIDs = a.loading.get_CIDs(['training','leaderboard','testset']) 
dream_CID_dilutions = a.loading.get_CID_dilutions(['training','leaderboard','testset'])


# In[11]:


# Load the DREAM SMILES strings
dream_smiles = a.get_dream_smiles(dream_CIDs) 


# In[12]:


# Get isomeric SMILES from DREAM dataset
isomeric_CIDs = a.find_isomers(dream_CIDs,dream_smiles)


# In[13]:


# Show the isomeric pairs from the DREAM challenge
if a.HAS_OBABEL:
   a.show_isomers(isomeric_CIDs,dream_CIDs,dream_smiles)


# In[14]:


# Load DREAM perceptual data
dream_perceptual_data = a.loading.load_perceptual_data(['training','leaderboard','testset'])
Y_dream = a.dream.make_Y(dream_perceptual_data)
# We must also have an imputed version that we can use as input to the algorithm since it will not work with NaNs.  
Y_dream_imputed = a.dream.make_Y(dream_perceptual_data, imputer='median')


# In[15]:


isomeric_index = 0 # The first isomeric pair, which should be racemic vs L-something
a.plot_isomer_ratings(dream_perceptual_data,isomeric_CIDs[isomeric_index],('racemic','L'))


# In[16]:


shadmany_smiles = a.load_other_smiles(shadmany=True)
shadmany_data = a.load_data('shadmany')


# In[17]:


shadmany_data


# ## Compute DREAM molecule features and place into a data frame

# In[18]:


# Compute Morgan features for DREAM molecules
all_smiles = list(set(dream_smiles + shadmany_smiles))
morgan_sim_all = a.smiles_to_morgan_sim(all_smiles,all_smiles)


# In[19]:


# Compute NSPDK features for DREAM molecules
if a.HAS_OBABEL:
    nspdk_all = a.smiles_to_nspdk(all_smiles)
else:
    nspdk_all = []


# In[20]:


# Compute Dragon or Mordred features for DREAM molecules
if a.USE_DRAGON:
    dragon_all = a.smiles_to_dragon(all_smiles)
else:
    dragon_all = a.smiles_to_mordred(all_smiles)


# In[21]:


# Combine all DREAM molecular features into one dataframe
molecular_data_all = dragon_all.join(nspdk_all).join(morgan_sim_all).astype('float')


# In[22]:


molecular_data_dream = molecular_data_all.loc[dream_smiles]
# Reindex by the CIDs instead of the SMILES strings
assert list(molecular_data_dream.index)==dream_smiles
molecular_data_dream.index = dream_CIDs


# In[23]:


# Compute the final DREAM molecular features matrix.  
X_dream,good1,good2,means,stds,imputer = a.dream.make_X(molecular_data_dream,dream_CID_dilutions)


# In[24]:


dummy_intensity = -3.0
shadmany_CIDs = [shadmany_data.loc[smile]['Pubchem ID #'] for smile in shadmany_smiles]
shadmany_CID_dilutions = [(shadmany_data.loc[smile]['Pubchem ID #'],dummy_intensity) for smile in shadmany_smiles]
molecular_data_shadmany = molecular_data_all.loc[shadmany_smiles]
assert list(molecular_data_shadmany.index) == shadmany_smiles
molecular_data_shadmany.index = shadmany_CIDs
X_shadmany = a.dream.make_X(molecular_data_shadmany,shadmany_CID_dilutions,
                            good1=good1,good2=good2,means=means,stds=stds)[0]
# Reorder molecules to match the order in the spreadsheet
X_shadmany = X_shadmany.loc[[(cid,dummy_intensity) for cid in shadmany_data['Pubchem ID #']]]
# Confirm that the order matches
assert list(X_shadmany.index.get_level_values('CID')) == list(shadmany_data['Pubchem ID #'])


# ## Fit model to DREAM data

# In[25]:


# Model for DREAM data
rfs = {}
n_estimators = 25
these_descriptors = a.descriptors[:1] # Just intensity
Y_dream_mean = Y_dream.stack('Descriptor').mean(axis=1).unstack('Descriptor')
for i,descriptor in enumerate(these_descriptors):
    print("%d. Fitting model for %s..." % (i+1,descriptor))
    rfs[descriptor] = a.RandomForestRegressor(n_estimators=n_estimators)
    valid = Y_dream_mean[descriptor].notnull()
    rfs[descriptor].fit(X_dream.loc[valid].as_matrix(),Y_dream_mean[descriptor].loc[valid].as_matrix())


# In[26]:


# Check model quality by cross-validation on the DREAM data.  
from sklearn.model_selection import GroupShuffleSplit
x = X_dream.loc[valid]
y = Y_dream_mean[descriptor].loc[valid]
groups = x.index.get_level_values('CID')
ss = GroupShuffleSplit(n_splits=3)
for train,test in ss.split(x,groups=groups):
    rfs['Intensity'] = a.RandomForestRegressor(n_estimators=25,random_state=0)
    rfs['Intensity'].fit(x.iloc[train].as_matrix(),y.iloc[train].as_matrix())
    predicted = rfs['Intensity'].predict(x.iloc[test].as_matrix())
    observed = y.iloc[test]
    print(a.np.corrcoef(predicted,observed)[0,1])


# ## Compute predictions and display

# In[27]:


a.compare_smiles_lengths(dream_smiles,shadmany_smiles,['DREAM','Enantiomers']);


# In[28]:


a.compare_molecular_weights(molecular_data_dream,molecular_data_shadmany,['DREAM','Enantiomers']);


# In[29]:


predictions = a.make_predictions(rfs,X_shadmany,['Intensity'])


# In[30]:


plus_pred = predictions.iloc[0::2]['Intensity'].values
minus_pred = predictions.iloc[1::2]['Intensity'].values


# In[31]:


a.plt.scatter(plus_pred,minus_pred)
a.plt.plot([0,100],[0,100],'--')
a.plt.xlabel('(+) intensity prediction')
a.plt.ylabel('(-) intensity prediction')
a.plt.show()


# In[32]:


plus_thresh = shadmany_data.iloc[0::2]['Normalized Detection Threshold'].values
minus_thresh = shadmany_data.iloc[1::2]['Normalized Detection Threshold'].values
plus_thresh = a.np.clip(plus_thresh,1e-8,1e8)
minus_thresh = a.np.clip(minus_thresh,1e-8,1e8)


# In[33]:


a.plt.scatter(plus_thresh,minus_thresh)
a.plt.plot([1e-7,1e9],[1e-7,1e9],'--')
a.plt.xlim(1e-7,1e9)
a.plt.ylim(1e-7,1e9)
a.plt.xscale('log')
a.plt.yscale('log')
a.plt.xlabel('(+) detection threshold (ppb)')
a.plt.ylabel('(-) detection threshold (ppb)')
a.plt.show()


# In[34]:


delta_pred = plus_pred - minus_pred
delta_thresh = a.np.log10(minus_thresh / plus_thresh)
inds = delta_thresh.argsort()
delta_pred = delta_pred[inds]
delta_thresh = delta_thresh[inds]
negative = delta_thresh < 0
delta_thresh[negative] *= -1
delta_pred[negative] *= -1
a.plt.scatter(delta_pred,delta_thresh)
a.plt.xlabel('Predicted intensity difference (+) vs (-)')
a.plt.ylabel('Actual detection threshold difference\n in log units (+) vs (-)')
a.plt.plot([0,0],[0,8],'--')
a.plt.xlim(-1,1)
a.plt.show()


# In[35]:


print("Correlation coefficient R = %.3f" % a.np.corrcoef(delta_pred,delta_thresh)[0,1])


# In[36]:


a.plt.scatter(plus_thresh,plus_pred,color='r',label='(+)')
a.plt.scatter(minus_thresh,minus_pred,color='b',label='(-)')
a.plt.xscale('log')
a.plt.xlim(1e-9,1e9)
a.plt.xlabel('Detection Threshold (ppb)')
a.plt.ylabel('Predicted Intensity')
a.plt.legend()
a.plt.show()
print("Correlation between predicted intensity and actual detection thresholds is R=%.3f" % \
      a.np.corrcoef(a.np.concatenate((plus_thresh,minus_thresh)),a.np.concatenate((plus_pred,minus_pred)))[0,1])


# In[ ]:




