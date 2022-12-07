#!/usr/bin/env python
# coding: utf-8

# ## Preliminaries

# In[1]:


# TODO: Recursive feature elimination to improve model and identify features
# TODO: Use harmonic mean (within pair) detection threshold as predictor


# In[2]:


#Gets the updates from the development files that are imported
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Imports the pands library, the math library, and the init class python file
#Last line: updates matlab library that is used in init file
import Utils as model_helpers
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrfume 
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# ## Start of Project

# In[8]:


# Loads in the data
load_smiles_strings = model_helpers.load_other_smiles(coleman=True)
enantiomer_data = model_helpers.load_data("coleman")


# In[9]:


# Groups each pair of enantiomers by making them have the same number associated with column "N"
# Takes each pair of enantiomers and computes the ratio of the Normalized Detection Thresholds between the two
# Sets the Normalized Detection Thresholds to be of the same type to avoid type errors later on
enantiomer_data['N'] = np.arange(0, enantiomer_data.shape[0]/2, 0.5).astype(int)
enantiomer_data['Normalized Detection Threshold'] = enantiomer_data['Normalized Detection Threshold'].astype('float')
enantiomer_data.head()


# In[10]:


# Adding all new smiles strings from "Other SMILES" column to official "SMILES String" column
enantiomer_data["SMILES String"] = enantiomer_data["Other SMILES"].combine_first(enantiomer_data["SMILES String"]) 


# In[11]:


# Searching for a specific molecule in the dataframe. Checking that SMILES Strings differ
enantiomer_data[enantiomer_data['Molecule Name'].str.contains('lina')]


# ## Cutting the Dataframe in Half

# In[13]:


# Take the absolute log values and harmonic values for each enantiomeric pair
half_log_abs = enantiomer_data.groupby('N').apply(model_helpers.log_abs)
half_det = enantiomer_data.groupby('N').apply(model_helpers.harmonic)


# In[14]:


# Creates a new data frame with just one odorant of each enantiomeric pair from the original dataset 
# adds the absolute value and detection threshold value for remaining odorants from enantiomering pair
half_enantiomer_data = enantiomer_data.iloc[::2].copy()
half_enantiomer_data.loc[:, 'log_abs'] = half_log_abs.values
half_enantiomer_data.loc[:, 'det'] = half_det.values


# In[15]:


# This line makes sure that the rest of the exsisting null values are equal in the new data frame and in the new data frame's 'log_abs' column
assert half_log_abs.isnull().sum() == half_enantiomer_data['log_abs'].isnull().sum()


# In[16]:


# This line checks that log_abs and det columns were added properly
half_enantiomer_data.head()


# In[17]:


# Gets rid of all the invalid SMILES Strings, specifically the duplicates because we don't want to count their perceptual features twice and the "nan" values 
half_enantiomer_data = half_enantiomer_data.drop_duplicates(subset=['SMILES String'])
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['SMILES String'].str.contains('NaN', na=True)]
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['SMILES String'].str.contains('nan', na=True)]


# In[18]:


# Assert statement to ensure that we only have unqiue smiles strings 
assert half_enantiomer_data['SMILES String'].shape == half_enantiomer_data['SMILES String'].unique().shape, "Number of SMILES strings should equal number of unique SMILES strings at this stage"


# In[19]:


# Assert that there are no more nan values in the smiles string column
assert sum(half_enantiomer_data['SMILES String']=='nan') == 0, "There should be no NaN SMILES strings at this point"


# In[20]:


# Gets rid of the rows with a null log_abs value
half_enantiomer_data = half_enantiomer_data[~half_enantiomer_data['log_abs'].isnull()]


# In[21]:


# Assert that there are no more log_abs of det values with the value null
assert not sum(half_enantiomer_data['log_abs'].isnull())
assert not sum(half_enantiomer_data['det'].isnull())


# ## 3rd Party Embeddings

# In[22]:


# Loads in embeddings from 3rd party model to use as features
gme = np.load('enantiomer-embeddings-for-rick.npz', allow_pickle=True) # Load the file
gme = gme['embeddings'].item() # Extract the data
gme = {k: v.squeeze() for k, v in gme.items()} # Flatten the arrays
gme_df = pd.DataFrame(gme).T  # Turn into a dataframe


# In[23]:


gme_df.head()


# In[25]:


# Make copy of original data and set the index to match that of the gme model
half_enantiomer_data_copy = half_enantiomer_data
half_enantiomer_data_copy = half_enantiomer_data_copy.set_index("SMILES String")


# In[26]:


# Keep the columns in the gme df that match the index names in original dataset
common_index = half_enantiomer_data_copy.index.intersection(gme_df.index)
half_enantiomer_data_copy = half_enantiomer_data_copy.loc[common_index]
gme_df = gme_df.loc[common_index]


# In[27]:


# Combine original dataset with gme df
g_model_embeddings = half_enantiomer_data_copy.join(gme_df, how="inner")


# In[28]:


# Reset the index to be "Moecule Name"
g_model_embeddings = g_model_embeddings.set_index("Molecule Name")


# In[29]:


g_model_embeddings.head()


# In[35]:


#assert ((g_model_embeddings.iloc[:,11:].var() <= 0).sum() == 0), "This should be 0 if not, get rid of columns with 0 varience"


# ## Computing the Features

# In[30]:


# Remove line separaters
half_enantiomer_data["SMILES String"] = half_enantiomer_data["SMILES String"].apply(lambda x : x.replace("\\n", "") and x.replace("\\r", ""))
half_enantiomer_data["SMILES String"] = half_enantiomer_data["SMILES String"].apply(lambda x : x.replace("\\n", ""))


# In[31]:


# Calculate the mordred features
mordred_data = model_helpers.calculate_features(half_enantiomer_data, "mordred")


# In[44]:


# Calculate the morgan features
morgan_data = model_helpers.calculate_features(half_enantiomer_data, "morgan")


# In[45]:


#zero_var_cols = [mordred_data[col] for col in mordred_data.iloc[:,11:] if (mordred_data[col].var() > 0) == True]
#mordred_data.drop(columns)


# In[46]:


# Dataframe with molecules that have mordred and morgan features computed
common_index = mordred_data.index.intersection(morgan_data.index)
mordred_data = mordred_data.loc[common_index]
morgan_data = morgan_data.loc[common_index]


# In[47]:


# Reset index
mordred_data.set_index('Molecule Name').head().iloc[:, 10:];
morgan_data.set_index('Molecule Name').head().iloc[:, 10:];


# In[48]:


# Data frame that has both the mordred and morgan features
both = mordred_data.join(morgan_data.iloc[:,10:], how="inner", rsuffix='morg_')
both.head()


# In[49]:


#Need to drop var columns
print(both.var().max())
print(both.var().min())


# In[50]:


# Gets all Mordred or Mogan features that have numeric values and not Null values
# Joins the final mordred and morgan features 
finite_mordred = model_helpers.finite_features(mordred_data)
finite_morgan = model_helpers.finite_features(morgan_data)
both_features = finite_mordred | finite_morgan


# ## PCA and VCD (Could delete since there were no matches)

# In[51]:


# Read in pca and vcd dataframe to use as features
pca_vcd_values = pd.read_csv("pca_vcd_values.csv")


# In[52]:


# Set index to Molecule Name
pca_vcd_values = pca_vcd_values.set_index('化合物名')


# In[35]:


pca_vcd_values.head()


# In[36]:


# Copy original dataframe
half_enantiomer_data_copy_two = half_enantiomer_data
half_enantiomer_data_copy_two;


# In[37]:


common_index = half_enantiomer_data_copy.index.intersection(pca_vcd_values.index)
half_enantiomer_data_copy = half_enantiomer_data_copy.loc[common_index]
pca_vcd_features = pca_vcd_values.loc[common_index]


# In[38]:


pca_vcd = half_enantiomer_data_copy.join(pca_vcd_features, how="inner")


# In[39]:


pca_vcd


# ## Creating the Models

# In[42]:


# Illustrate the magnitude differences across enantiomeric pairs in the dataset
model_helpers.fold_difference_of_enantiomers(half_enantiomer_data)


# In[53]:


# Gets the appropriate parameter values for mordred model
# Gets the valid features (not null values) from feature data frame and the log_abs values from the feature dataframe
X_morded = mordred_data[finite_mordred]
y = mordred_data['log_abs']
X_morded = X_morded[y < 10]
y_mordred = y[y < 10]
Xn_mordred = pd.DataFrame(StandardScaler().fit_transform(X_morded), index=X_morded.index, columns=X_morded.columns)


# In[54]:


# Gets the appropriate parameter values for Morgan model
# Gets the valid features (not null values) from feature data frame and the log_abs values from the feature dataframe
x_morgan = morgan_data[finite_morgan]
y_morgan = morgan_data["log_abs"]
x_morgan = x_morgan[y_morgan < 10]
y_morgan = y_morgan[y_morgan < 10]
Xn_morgan = pd.DataFrame(StandardScaler().fit_transform(x_morgan), index=x_morgan.index, columns=x_morgan.columns)


# In[60]:


# Model for Morgan dad
model_helpers.create_model(Xn_morgan, y_morgan)


# In[85]:


model_helpers.cross_val(Xn_morgan, y_morgan)


# In[59]:


# Model for Mordred data
model_helpers.create_model(Xn_mordred, y_mordred)


# In[86]:


model_helpers.cross_val(Xn_mordred, y_mordred)


# In[70]:


x_gme = g_model_embeddings.iloc[:,11:]
y_gme = g_model_embeddings["log_abs"]
Xn_gme = pd.DataFrame(StandardScaler().fit_transform(x_gme), index=x_gme.index, columns=x_gme.columns)


# In[71]:


model_helpers.create_model(Xn_gme, y_gme)


# ### Liyah, stop here.  I can explain what I am doing in the next section another time.

# In[87]:


from sklearn.feature_selection import RFE, RFECV
svr = SVR(C=10, kernel='linear')
rfe = RFE(svr, n_features_to_select=1, step=10)
#rfe.fit(Xn, y)


# In[62]:


rfe.ranking_.max()


# In[63]:


svr = SVR(C=10, kernel='rbf')
ns = range(1, 109, 1)
rs = pd.Series(index=ns, dtype=float)
for n in tqdm(ns):
    Xn_ = Xn[Xn.columns[rfe.ranking_ <= n]]
    y_predict = cross_val_predict(svr, Xn_, y, cv=LeaveOneOut(), n_jobs=-1)
    rs[n] = np.corrcoef(y, y_predict)[0, 1]


# In[ ]:


rs.plot()


# In[ ]:


Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
Xn_ = Xn[Xn.columns[rfe.ranking_ <= 10]]
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn_, y)
    y_predict_in = clf.predict(Xn_)
    y_predict_out = cross_val_predict(clf, Xn_, y, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y, y_predict_in)[0]
    rs_out[C] = pearsonr(y, y_predict_out)[0]
    rhos_out[C] = spearmanr(y, y_predict_out)[0]


# In[ ]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[ ]:


Xn_ = Xn[Xn.columns[rfe.ranking_ <= 10]]
rfr = RandomForestRegressor(n_estimators=100)
y_predict = cross_val_predict(rfr, Xn_, y, cv=LeaveOneOut(), n_jobs=-1)
np.corrcoef(y, y_predict)[0, 1]


# In[ ]:


rfr.fit(Xn_, y)
pd.Series(rfr.feature_importances_, Xn_.columns).sort_values(ascending=False).head(25)


# In[83]:


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
sns.set_style('whitegrid')
svr = SVR(C=10, kernel='rbf')
svr.fit(Xn_mordred, y_mordred)
y_predict = cross_val_predict(svr, Xn_mordred, y_mordred, cv=LeaveOneOut(), n_jobs=-1)
y_predict = np.clip(y_predict, 0, np.inf)
plt.figure(figsize=(5, 5))
plt.scatter(y_mordred, y_predict, alpha=0.3)
#maxx = y.max()*1.01
#plt.plot([0, maxx], [0, maxx], '--')
plt.plot([0, 4], [0, 4], '--')
plt.xlim(0, 4)
plt.ylim(0, 4)
#plt.xlim(0.9, maxx)
#plt.ylim(0.9, maxx)
plt.title('R = %.2g' % np.corrcoef(y_mordred, y_predict)[0, 1])
#ticks = range(5)
#plt.xticks(ticks)
#plt.yticks(ticks)
plt.xticks([1,2,3,4], ["1x", "10x", "100x", "1000x"])
plt.yticks([1,2,3,4], ["1x", "10x", "100x", "1000x"])
# plt.xticks(ticks, ['%d' % 10**x for x in ticks])
# plt.yticks(ticks, ['%d' % 10**x for x in ticks])
plt.xlabel('Actual Detection Threshold Ratio')
plt.ylabel('Predicted Detection\nThreshold Ratio')


# In[61]:


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
sns.set_style('whitegrid')
svr = SVR(C=10, kernel='rbf')
svr.fit(Xn_morgan, y_morgan)
y_predict = cross_val_predict(svr, Xn_morgan, y_morgan, cv=LeaveOneOut(), n_jobs=-1)
y_predict = np.clip(y_predict, 0, np.inf)
plt.figure(figsize=(5, 5))
plt.scatter(y_morgan, y_predict, alpha=0.3)
plt.plot([0, 4], [0, 4], '--')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.title('R = %.2g' % np.corrcoef(y_morgan, y_predict)[0, 1])
plt.xticks([1,2,3,4], ["1x", "10x", "100x", "1000x"])
plt.yticks([1,2,3,4], ["1x", "10x", "100x", "1000x"])
plt.xlabel('Actual Detection Threshold Ratio')
plt.ylabel('Predicted Detection\nThreshold Ratio')


# In[ ]:


sns.heatmap(svr.support_vectors_, cmap='RdBu_r')


# In[ ]:


abraham = pyrfume.load_data('abraham_2011/abraham-2011-with-CIDs.csv')
abraham = abraham.dropna(subset=['SMILES'])
from pyrfume.features import smiles_to_mordred
abraham_mordred = smiles_to_mordred(abraham['SMILES'].values)


# In[ ]:


#X = abraham_mordred.join(abraham.set_index('SMILES').drop(['Substance', 'Group'], axis=1), how='inner', rsuffix='abr_')
abe_ok = ['MW',
 'Log (1/ODT)',
 'E',
 'S',
 'A',
 'B',
 'L',
 'V',
 'M',
 'AL',
 'AC',
 'ES',
 'C1',
 'C1AC',
 'C1AL',
 'HS',
 'C2',
 'C2Al',
 'C2AC']
X = abraham_mordred.join(abraham.set_index('SMILES')[['Log (1/ODT)']], how='inner', rsuffix='abr_')
y = X['Log (1/ODT)']
X = X.astype('float').drop('Log (1/ODT)', axis=1)
#finite_all = X.dropna(axis=1).columns.intersection(finite_mordred)
X = X[finite_mordred].fillna(0)
X = X.dropna(axis=1)
Xn = X.copy()
Xn[:] = StandardScaler().fit_transform(X)


# In[ ]:


Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn, y)
    y_predict_in = clf.predict(Xn)
    y_predict_out = cross_val_predict(clf, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y, y_predict_in)[0]
    rs_out[C] = pearsonr(y, y_predict_out)[0]
    rhos_out[C] = spearmanr(y, y_predict_out)[0]


# In[ ]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[ ]:


svr = SVR(C=1, kernel='rbf')
svr.fit(Xn, y)
X_ = mordred_data[finite_mordred]; y_ = mordred_data['log_abs']
X_ = X_.join(half_coleman_data.set_index('SMILES String')['det'])
X_ = X_[y_ < 10]
y_ = y_[y_ < 10]
Xn_ = pd.DataFrame(StandardScaler().fit_transform(X_), index=X_.index, columns=X_.columns)
X_['det'] = svr.predict(Xn_.drop('det', axis=1))
#y_predict = cross_val_predict(svr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)


# In[ ]:


Xn_ = pd.DataFrame(StandardScaler().fit_transform(X_), index=X_.index, columns=X_.columns)
Cs = np.logspace(-3, 3, 13)
rs_in = pd.Series(index=Cs, dtype=float)
rs_out = pd.Series(index=Cs, dtype=float)
rhos_out = pd.Series(index=Cs, dtype=float)
for C in tqdm(Cs):
    svr = SVR(C=C, kernel='rbf')
    clf = svr
    #y_predict = cross_val_predict(rfr, Xn, y, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(Xn_, y_)
    y_predict_in = clf.predict(Xn_)
    y_predict_out = cross_val_predict(clf, Xn_, y_, cv=LeaveOneOut(), n_jobs=-1)
    y_predict_in = np.clip(y_predict_in, 0, np.inf)
    y_predict_out = np.clip(y_predict_out, 0, np.inf)
    rs_in[C] = pearsonr(y_, y_predict_in)[0]
    rs_out[C] = pearsonr(y_, y_predict_out)[0]
    rhos_out[C] = spearmanr(y_, y_predict_out)[0]


# In[ ]:


rs_in.plot(label='In-sample prediction R')
rs_out.plot(label='Out-of-sample prediction R')
rhos_out.plot(label=r'Out-of-sample prediction $\rho$')
plt.xscale('log')
plt.ylabel('Correlation\n(predicted vs observed)')
plt.xlabel('C (SVR hyperparameter)')
plt.legend(fontsize=10)


# In[ ]:




