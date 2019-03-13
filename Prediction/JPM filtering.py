
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 25)


# In[2]:


def clean_names(name_list):
    col_names = []
    for name in name_list:
        t = name.lower()
        if " " in t:
            c_t = t.replace(' ',"_")
        else:
            c_t = t
        col_names.append(c_t)
    return col_names


# In[3]:


def fill_time(df1):
    
    copy = df1.iloc[:,0].apply(lambda x: None if len(x) <= 12 else x)
    copy = copy.ffill()
    
    return copy
    
 


# In[4]:


''' 
The first step of preparing the dataframe: 
1. drop the na columns or rows after examnination 
2. clean up column names 

'''

def prep_df_1 (raw_df):
    temp = raw_df.dropna(axis = 1, how = 'all')
    names = temp.columns.tolist()
    col_names = clean_names(names)
    
    temp.columns = col_names
    
    # not the most ideal, but this way keeps the original idx in tact
    temp_2 = temp.dropna(axis = 0, thresh = 16)
    
    return temp_2
   
  


# In[5]:


def prep_df_2(raw):
    df1 =  prep_df_1(raw) #(58324, 20)
    dates = fill_time(df1)
    df1.insert(0, column = 'dates', value = pd.Series(dates))
    df1 = df1.dropna() # (58308, 22)
    df2 = df1.reset_index()
    df3 = df2.drop(columns = ['index', 'unnamed:_0'], axis = 1)
    
    return df3


# In[6]:


raw = pd.read_csv('chill.csv', thousands = "," , skiprows = 6)


# In[8]:


p1_df = prep_df_1(raw)
p2_df = prep_df_2(p1_df)
p2_df.head(4)


# ### convert dates to time stamps
# 

# In[9]:


def clean_time(p2_df):
    from datetime import datetime
    
    start_date,end_date = p2_df['dates'].str.split('-', 1).str
    p2_df.insert(0,'start_date', start_date)
    p2_df.insert(1, 'end_date', end_date)
    
    p2_df.start_date = p2_df.start_date.apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    p2_df.end_date = p2_df.end_date.apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    
    p2_df.drop(labels = ['dates'], axis = 1, inplace = True)
    
    return p2_df


# In[10]:


'''
clean_df is the df of the entire universe of bonds 

'''
clean_df = clean_time(p2_df)


# ### look at individual bonds spread over 10 day time period. If this bond spread is trading at its widest(cheapest) 5% level, over the course of 10 day, make this criteria 
# 

# In[12]:


def mark_gspd(clean_df, today_date, isin_list):
    check_gspd = []
    
    for isin in isin_list:
        if isin in np.array(clean_df['isin']):
            one_bond = clean_df[clean_df['isin'] == isin] 
            today_gspd = one_bond[one_bond['end_date'] == today_date]['g_spd'].values
            max_gspd = pd.to_numeric(one_bond['g_spd']).max()
            min_gspd = pd.to_numeric(one_bond['g_spd']).min()
            delta = max_gspd - min_gspd
        
        if today_gspd >= min_gspd + delta*0.95 and delta >= 10:
            check_gspd.append(1)
        elif today_gspd <= min_gspd + delta*0.05 and delta >= 10:
            check_gspd.append(-1)
        else:
            check_gspd.append(0)
            
    return check_gspd
    


# In[13]:


today_date = '2019-02-06'

# find out the date we want to check rich or cheap 
today_df = clean_df[clean_df.end_date == today_date]

# put all the isin into a list
isin_all = today_df['isin']  # len:5832

check_gspd = mark_gspd(clean_df, today_date, isin_all)

today_df['check_gspd'] = check_gspd


# In[14]:


# expensive bonds  
expensive = today_df[today_df['check_gspd'] == -1]

print("There are", expensive.shape[0], "expensive bonds.")


# In[16]:


# cheap bonds - good
cheap = today_df[today_df['check_gspd'] == 1]

print("There are", cheap.shape[0], "cheap bonds.")


# In[17]:


# middle 
between = today_df[today_df['check_gspd'] == 0]
print("There are", between.shape[0], "bonds.")


# ### Fit the issuer curve
# - compare the spread against the issuer curve, which is just all the bonds within the same issuer and if that individual bond is trading wide to the issuer curve it meets another criteria
# - issuer curve, y-axis: g_spread , x-axis: years_to_ maturity 

# In[55]:


ticker_groupby = clean_df.groupby(['ticker', 'end_date','isin'])['g_spd', 'years_to_mat'].mean() # shape:(58314, 2)


# In[76]:


np.random.seed(1729)

def func(x, m, c):
    return m*np.log(x)+ c 

def cal_curve(ticker_groupby):
    
    from scipy.optimize import curve_fit

    tickers = pd.unique(ticker_groupby.index.get_level_values(0))
    dates = pd.unique(ticker_groupby.index.get_level_values(1))
    ticker_temp = ticker_groupby.reset_index()
    
    g_spd_pred =[]
    diff = []
    idx = []
    
    for t in tickers:
        for d in dates:
            temp = ticker_temp[ticker_temp['ticker'] == t]
            sub_df = temp[temp['end_date'] == d]
            
            if sub_df.shape[0] > 2:
                sub_x = sub_df['years_to_mat'].values
                sub_y = sub_df['g_spd'].values
            
                popt, pcov = curve_fit(func, sub_x, sub_y)
                g_predict = func(sub_x, popt[0], popt[1])
                sub_diff = g_predict - sub_y
                g_spd_pred.append(g_predict)
                diff.append(sub_diff)
                idx.append(sub_df.index)
            
    g_pred = np.concatenate(g_spd_pred)
    g_diff = np.concatenate(diff)
    g_idx = np.concatenate(idx)
    
    return g_pred, g_diff, g_idx 
    
    


# In[77]:


g_pred, g_diff, g_idx = cal_curve(ticker_groupby)  


# In[78]:


result_df = ticker_groupby.iloc[g_idx]


# In[79]:


result_df['g_pred'] = g_pred


# In[80]:


result_df['g_diff'] = g_diff  # result_df.shape : (54806, 4)


# In[89]:


peer_df = result_df.reset_index() # peer_df.shape: (54806, 7), unique isin 5572


# - the total num of unique isins in the dataset is 5925 
# - after filtering the ones we cant fit the line with peers, there are 5572 unique isin in peer_df
# - the num of target date unique isin is 5832
# - the intersect between target date isin and peer isin is 5481
# - however, when we get all the peer isin info on the target date, we are missing 4 isins
# 

# In[91]:


def self_peer_filter(peer_df, target_isin, target_date):
    
    avail_isin = np.intersect1d(target_isin, peer_df['isin'].values) 
    
    prep_self = peer_df[peer_df['isin'].isin(avail_isin)]
    self_df = prep_self.groupby(['isin'])[['g_diff']].max().reset_index()
    self_df.columns = ['isin', 'g_diff_max']
    
    self_df['g_diff_min'] = prep_self.groupby(['isin'])[['g_diff']].min().values
    self_df['delta'] = self_df['g_diff_max'] - self_df['g_diff_min']
    self_df['perc_95'] = self_df.g_diff_min + self_df.delta.values*0.95
    self_df['bottom_5'] = self_df.g_diff_min + self_df.delta.values*0.05
   
    # making sure delta is >= 10
    delta_10 = self_df[self_df['delta']>=10]
    delta_isin = delta_10['isin'].values
    sub_peer = peer_df[peer_df['isin'].isin(delta_isin)]
    
    target_diff = sub_peer[sub_peer['end_date'] == target_date][['isin', 'g_diff']]  #1429 rows × 7 columns
    sub_self = self_df[self_df['isin'].isin(target_diff['isin'].values)]
    
    return target_diff, sub_self


# In[84]:


def get_results(target_diff, sub_self):
    c = target_diff['g_diff'].values - sub_self['perc_95'].values
    cheap =[1 if diff >0 else 0 for diff in c] 
    sub_self['cheap'] = cheap
    
    r = target_diff['g_diff'].values - sub_self['bottom_5'].values
    rich = [1 if diff < 0 else 0 for diff in r]
    sub_self['rich']= rich
    
    sub_self['target_day'] = target_diff['g_diff'].values
    
    return sub_self


# In[92]:


target_date = "2019-02-06"
target_isin = clean_df[clean_df['end_date'] == target_date]['isin'].values
#len(target_isin)


# In[95]:


target_diff, sub_self = self_peer_filter(peer_df, target_isin, target_date)


# In[96]:


result_df = get_results(target_diff, sub_self)


# In[102]:


result_cheap = result_df[result_df['cheap']==1]
print("There are",result_cheap.shape[0], "bonds are cheap." )


# In[103]:


result_rich = result_df[result_df['rich']==1]
print("There are",result_rich.shape[0], "bonds are rich." )

