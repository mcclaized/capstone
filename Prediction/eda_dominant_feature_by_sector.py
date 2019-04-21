#!/usr/bin/env python
# coding: utf-8

# - In-depth EDA: Dominant Feature by sector (through PCA)
# - Yilin Sun
# - latest version: 20190421

# ### Read data and preprocess

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# read data
df = pd.read_csv('vanguard_merge.csv')

# remove blanks from column names
df.columns = df.columns.str.strip().str.replace('-', '').str.replace(' ','').str.replace('&','')

# drop NA and irrelevant columns
df = df.dropna().drop(['ClassDetailCode','date','Country','ISIN','IssDt','Mty','Ticker'], axis=1)

# sort by sectors
df = df.sort_values(by='BCLASS3')

# descriptions
print('There are %d sectors in total.' % len(set(df.BCLASS3)))
print('namely:', [i for i in set(df.BCLASS3)])


# ### Define function for PCA

# In[3]:


def topN_pca(data, n, columnnames):
    """
    input: 
        data - dataframe including sectors & features
        n - number of PC to keep
        
    output:
        finalDf - dataframe of PC & corresponding sector
        variance_explained - percentage of total variance explained by each PC
        loading - loadings for each PC
    """
    # extract sectors
    y = data.loc[:,['BCLASS3']].values
    
    # extract and standardize features
    x = data.loc[:, df.columns[1:]].values
    x = StandardScaler().fit_transform(x)
    features = data.columns[1:]
    
    # pca
    pca = PCA(n_components=n)
    PCs = pca.fit_transform(x)
    
    # principal component dataframe
    pcDf = pd.DataFrame(data = PCs, columns = column_names)

    # display first n principal components and concatenate with sectors name
    finalDf = pd.concat([pcDf, df[['BCLASS3']]], axis = 1)
    
    # loadings dataframe
    loading = pd.DataFrame(pca.components_, columns=list(features))
    
    # dominant features for each PC
    top1_dominant_features = pd.DataFrame(features[np.argmax(np.abs(loading.values), axis=1)].values).T
    top1_dominant_features.columns = column_names
    
    # total variance explained by each PC
    variance_explained = pca.explained_variance_ratio_
    
    return finalDf, variance_explained, loading, top1_dominant_features


# In[4]:


# define: keep top N principal components
topN_pc = 2

# generate output dataframe's column names
column_names = ['PC '+str(i+1) for i in range(topN_pc)]
column_names2 = ['pc '+str(i+1) for i in range(topN_pc)]
column_names_var = ['Var_Explained_by_PC'+str(i+1) for i in range(topN_pc)]
column_names_dominant_features = ['Dominant_Feature_of_PC'+str(i+1) for i in range(topN_pc)]

# generate empty daraframe for pca results  
dominant_feature_bysector = pd.DataFrame(index=set(df.BCLASS3), columns=column_names)
variance_explained_bysector = pd.DataFrame(index=set(df.BCLASS3), columns=column_names2)

# begin pca by sector
for row_ind, sector in enumerate(set(df.BCLASS3)):
    
    # progress bar
    print('%d/18'%row_ind, sector)
    
    # pca for one sector
    data = df.loc[df.BCLASS3 == sector]
    pc, variance_explained, loading, dominant_feature = topN_pca(data, topN_pc, column_names)
     
    # append results from individual sector to master dataframes
    dominant_feature_bysector.iloc[row_ind] = dominant_feature.values
    variance_explained_bysector.iloc[row_ind] = np.around(variance_explained, decimals=3)
    
# cumulative variance explained
variance_explained_bysector['cumulativeVar'] = variance_explained_bysector.sum(axis=1)


# In[5]:


dominant_feature_bysector.to_csv('dominant_feature_bysector.csv')
variance_explained_bysector.to_csv('variance_explained_bysector.csv')


# In[9]:


results = pd.concat([dominant_feature_bysector, variance_explained_bysector], axis=1)
results = results.sort_values(by='cumulative_Var', ascending=False)
results.to_csv('merge_bysector.csv')
results.sort_values(by='PC 2')


# In[7]:


"""
def plot_top2_pc():
    # visualize
    fig = plt.figure(figsize = (16,16))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Top 2 Component from PCA', fontsize = 20)
    
    sectors = set(df.BCLASS3)
#colors = list(range(len(set(df.BCLASS3))))
for sector, color in zip(sectors,colors):
    indicesToKeep = finalDf['BCLASS3'] == sector
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], 
               finalDf.loc[indicesToKeep, 'pc2'],
               s = 2) #c = color, 

ax.legend(sectors)
ax.grid()




# visualize
fig = plt.figure(figsize = (16,16))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Top 2 Component from PCA', fontsize = 20)

sectors = set(df.BCLASS3)
for sector, color in zip(sectors,colors):
    indicesToKeep = finalDf['BCLASS3'] == sector
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], 
               finalDf.loc[indicesToKeep, 'pc2'],
               s = 2) 

ax.legend(sectors)
ax.grid()
"""


# In[ ]:




