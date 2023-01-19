#!/usr/bin/env python
# coding: utf-8

# ## Predicting Divergence Notebook

# ## Preliminaries

# In[5]:


# TODO: Recursive feature elimination to improve model and identify features
# TODO: Use harmonic mean (within pair) detection threshold as predictor


# In[6]:


#Gets the updates from the development files that are imported
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


#Imports the pands library, the math library, and the init class python file
#Last line: updates matlab library that is used in init file
import Utils as model_helpers
import pandas as pd


# ## Load Data

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




