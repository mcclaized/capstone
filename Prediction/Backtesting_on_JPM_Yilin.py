#!/usr/bin/env python
# coding: utf-8

# - Backtestting
# - Yilin Sun
# - last updated 2019-04-29
# 
# #### Four period
# - 2018-04-10 to 2018-04-30
# - 2018-07-12 to 2018-08-01
# - 2018-10-15 to 2018-11-02
# - 2019-01-16 to 2019-02-05

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# read data
df = pd.read_csv('vanguard_merge.csv')

# remove blanks from column names
df.columns = df.columns.str.strip().str.replace('-', '').str.replace(' ','').str.replace('&','')

# keep only relevant columns
curve = df[['date', 'BCLASS3', 'ISIN', 'GSpd', 'YearstoMat', 'Ticker', 'SPRatingNum', 'PxClose']]
curve = curve.dropna()

# transform Year to Maturity(YTM) using exponential transformation preparing for curve fitting
shortterm = 10
longterm = 40
ytm = curve.YearstoMat.values
curve['yr5'] = (1 - np.exp(- (1/shortterm) * ytm))/ytm
curve['yr10'] = (1 - np.exp(- (1/longterm) * ytm))/ytm


# In[3]:


### Step 1: Define G-spread Filtering
def GSpdFiltering(input_date, curve):
    
    curve = curve.sort_values(by=['date', 'ISIN'])[['GSpd', 'ISIN', 'date']]
 
    # get the set of dates
    dates = np.unique(curve.date)
    today_ind = np.where(dates == input_date)[0][0]
    
    # lookahead bias - use yesterday's data to compare with the previous 30 days
    yesterday = dates[today_ind-1]
    
    # get the dataframe for the previous 30 days + yesterday
    clean = pd.DataFrame()
    for day in dates[(today_ind-31):(today_ind)]: 
        clean = clean.append(curve[curve.date == day])
        
    # base - previous 30 days; queryday - yesterday
    queryday = clean[clean.date == yesterday][['GSpd', 'ISIN']]
    clean = clean[['GSpd', 'ISIN']]
    
    # calculate upper and lower 5 perc
    result = pd.DataFrame()
    result['ISIN'] = clean.groupby('ISIN').max().index
    result['maxi'] = clean.groupby('ISIN').max().values
    result['mini'] = clean.groupby('ISIN').min().values
    result['delta'] = result.maxi - result.mini
    result['upper95'] = result.maxi - result.delta*0.05
    result['bottom5'] = result.mini + result.delta*0.05
    
    # keep issuers with at least 10bp spread
    result = result.query('delta >= 10')
    
    # merge previous 30 days benchmark with queryday spread
    result = pd.merge(result, queryday, on = 'ISIN', how = 'inner')
    
    # find the rich and cheap bonds: above 95pc - cheap bonds; below 5pc - rich bonds
    result['cheap'] = result.GSpd - result.upper95
    result['rich'] = result.bottom5 - result.GSpd
    
    # filter out rich and cheap bonds
    richbonds = result.query('rich > 0').ISIN
    cheapbonds = result.query('cheap > 0').ISIN
    
    return richbonds, cheapbonds


# In[4]:


### Step 2: Define Issuer Curve Filtering
def issuerCurve(input_date, curve):
    
    # sort values - not necessary but suggested
    curve = curve.sort_values(by=['date', 'Ticker', 'ISIN'])
    
    # initialize
    distance = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])
    issuers = np.unique(curve.Ticker)
    dates = np.unique(curve.date)
    
    # index location of the query day
    today_ind = np.where(dates == input_date)[0][0]
    yesterday = dates[today_ind-1]
    
    # calculate distance matrix for all bonds all 30 days
    for i, day in enumerate(dates[(today_ind-31):(today_ind)]): 
        if i%5 == 0:
            print('issuer', day)
        oneday = curve[curve.date == day]
 
        for issuer in issuers:
            oneissuer = oneday[oneday.Ticker == issuer]
            
            # get y
            y = oneissuer.GSpd.values
            
            # check if this sector has more than 5 bonds/day
            if len(y) > 5:
                
                # get Xs
                x = np.array((oneissuer.yr5, oneissuer.yr10)).T
                
                # fit linear regresion on OAS, 5yr, 10yr
                lr = linear_model.LinearRegression().fit(x, y)

                # prediction
                ypred = lr.predict(x)
                 
                # distances for each bond at one day
                dist = y - ypred
                
                # append the distance to output dataframe
                oneissuer['dist'] = dist
                distance = distance.append(oneissuer[['date', 'ISIN', 'dist']])
                
    # extract query day
    queryday = distance[distance.date == yesterday][['ISIN', 'dist']]
    base = distance
    
    # save final results
    result = pd.DataFrame()
    result['ISIN'] = base.groupby('ISIN').max()['dist'].index
    result['maxi'] = base.groupby('ISIN').max()['dist'].values
    result['mini'] = base.groupby('ISIN').min()['dist'].values
    result['delta'] = result.maxi - result.mini
    result['upper95'] = result.maxi - result.delta*0.05
    result['bottom5'] = result.mini + result.delta*0.05
    
    # keep issuers with at least 10bp spread
    result = result.query('delta >= 10')
    
    # merge 
    result = pd.merge(result, queryday, on = 'ISIN', how = 'inner').drop_duplicates()
    
    # find upper and lower 5pc
    result['cheap'] = result.dist - result.upper95
    result['rich'] = result.bottom5 - result.dist
    
    # filter out rich and cheap bonds
    richbonds = result.query('rich > 0').ISIN
    cheapbonds = result.query('cheap > 0').ISIN
    
    return result, richbonds, cheapbonds 


# In[5]:


### Step 3: Define Sector Curve Filtering
def sectorCurve(input_date, curve):
    
    # sort values - not necessary but suggested
    curve = curve.sort_values(by=['date', 'BCLASS3', 'ISIN'])
    
    # initialize
    distance = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])
    
    # unique issuer, dates, sprating
    sectors = np.unique(curve.BCLASS3)
    dates = np.unique(curve.date)
    sprating = np.unique(curve.SPRatingNum.values)
    
    # remove NAs from rating list
    ratings = (sprating[~np.isnan(sprating)])
    
    # index location of the query day
    today_ind = np.where(dates == input_date)[0][0]
    
    # lookahead bias - use yesterday's data to compare with the previous 30 days
    yesterday = dates[today_ind-1]
    
    # calculate distance matrix for all bonds all 30 days
    for i, day in enumerate(dates[(today_ind-31):(today_ind)]):
        
        if i%5 == 0:
            print('sector', day)
        
        oneday = curve[curve.date == day]
        
        for sector in sectors:
            onesector = oneday[oneday.BCLASS3 == sector]
             
            for rating in ratings:
                onerating = onesector[onesector.SPRatingNum == rating]
                 
                # get x, y
                y = onerating.GSpd.values
            
                # check if this issuer has >= 6 bonds/day
                if len(y) > 5:
                   
                    x = np.array((onerating.yr5, onerating.yr10)).T
                     
                    # fit linear regresion on OAS, 5yr, 10yr
                    lr = linear_model.LinearRegression().fit(x, y)

                    # prediction
                    ypred = lr.predict(x)

                    # distances for each bond at one day
                    dist = y - ypred
                    
                    # append to one sector distance
                    onerating['dist'] = dist
                    distance = distance.append(onerating[['date', 'ISIN', 'dist']])
                    
    # split into the first 30 days and the query day
    queryday = distance[distance.date == yesterday][['ISIN', 'dist']]
    base = distance
    
    # save final results
    result = pd.DataFrame()
    result['ISIN'] = base.groupby('ISIN').max()['dist'].index
    result['maxi'] = base.groupby('ISIN').max()['dist'].values
    result['mini'] = base.groupby('ISIN').min()['dist'].values
    result['delta'] = result.maxi - result.mini
    result['upper95'] = result.maxi - result.delta*0.05
    result['bottom5'] = result.mini + result.delta*0.05
    
    # merge 
    result = pd.merge(result, queryday, on = 'ISIN', how = 'inner').drop_duplicates()
    
    # keep issuers with at least 10bp spread
    result = result.query('delta >= 10')
    
    # find the rich and cheap bonds: above 95pc - cheap bonds; below 5pc - rich bonds
    result['cheap'] = result.dist - result.upper95
    result['rich'] = result.bottom5 - result.dist
    
    # filter out rich and cheap bonds
    richbonds = result.query('rich > 0').ISIN
    cheapbonds = result.query('cheap > 0').ISIN
    
    return result, richbonds, cheapbonds


# In[6]:


### Step 4: Define Merge Results
def merge_result(spd, iss, sec, curve):
    spd = spd.to_frame().reset_index()
    iss = iss.to_frame().reset_index() 
    sec = sec.to_frame().reset_index() 
    
    #merge
    output = pd.merge(spd, iss, on='ISIN', how='left')
    output = pd.merge(output, sec, on='ISIN', how='left')
    
    output.columns = ['spd', 'ISIN', 'iss', 'sec']
    output = output[['ISIN', 'spd', 'iss', 'sec']]
    
    #fill na with 0 
    output.spd = 1
    output.sec = output.sec/output.sec
    output.iss = output.iss/output.iss
    output = output.fillna(0)
    
    # get ranking & keep only rank>1
    output['ranking'] = output[['spd','iss','sec']].sum(axis=1)
    output = output.loc[output.ranking > 1].sort_values(by='ranking', ascending=False)
    
    # add sector and issuer info to output
    output = pd.merge(output, curve[['ISIN', 'BCLASS3', 'Ticker']], 
                                      on='ISIN', how='left').drop_duplicates(subset='ISIN')
    
    return output


# In[7]:


### Step 5: Define function - Merge Results from Three Filters
def rich_cheap(input_date, curve):
    
    # 1.G-Spread filtering
    rich_spd, cheap_spd = GSpdFiltering(input_date, curve)

    # 2.issuer curve
    result_iss, rich_iss, cheap_iss = issuerCurve(input_date, curve)

    # 3.sector curve
    result_sec, rich_sec, cheap_sec = sectorCurve(input_date, curve)

    # 4.merge all three
    richbonds = merge_result(rich_spd, rich_iss, rich_sec, curve)
    cheapbonds = merge_result(cheap_spd, cheap_iss, cheap_sec, curve)

    print('Merging results finished.')
    return richbonds, cheapbonds


# #### test: Filters defined. now get the results...

# In[8]:


"""
%%time

# test recommendations on one day - need 12~13min/day
input_date = '2019-03-07'
testrich, testcheap = rich_cheap(input_date, curve)
"""


# ### Calculate one-week return for each bond
# - Assuming one-week (5 trade days) holding period..

# In[9]:


### Step 6: Define One_week_returns for all bonds if buy on this day

def weekly_returns_on_a_day(start_date, curve):
    # get end date
    sorted_dates = curve.date.drop_duplicates().reset_index(drop=True).values
    end_date = sorted_dates[np.where(sorted_dates == start_date)[0] + 5][0]
    
    # all bonds return for the next week
    subset = curve[['date', 'ISIN', 'PxClose']]

    # get start and end day PxClose
    start_date_gsp = subset.loc[subset.date == start_date]
    end_date_gsp = subset.loc[subset.date == end_date]
    gsp = pd.merge(start_date_gsp, end_date_gsp, on='ISIN')
    gsp.columns = ['date_start', 'ISIN', 'PxClose_start', 'date_end', 'PxClose_end']

    # calculate one-week return
    gsp['one_week_returns']=(gsp.PxClose_end - gsp.PxClose_start)/gsp.PxClose_start
    weekly_return = gsp[['ISIN', 'one_week_returns']]

    # append sector, issuer, rating
    output = pd.merge(weekly_return, curve[['ISIN','Ticker','BCLASS3','SPRatingNum']], on='ISIN').drop_duplicates()
    
    return output


# In[10]:


### Part 7: Define function to get performance(in terms of percentiles on one-week-returns) of recommended bonds
def get_percentile(isin, returns):
    
    # return of this bond
    isin_return = returns.one_week_returns[returns.ISIN==isin].values[0]
    
    sector_rating_perc = -1
    issuer_perc = -1
    
    if not pd.isna(isin_return):
        
        # sector rating-wise percentiel
        sector = returns.BCLASS3[returns.ISIN==isin].values[0]
        rating = returns.SPRatingNum[returns.ISIN==isin].values[0]
        sector_rating_all = returns.loc[returns.BCLASS3==sector].loc[returns.SPRatingNum==rating].one_week_returns.values
        sector_rating_all = np.array(sorted(sector_rating_all))
        sector_rating_perc = (np.where(sector_rating_all==isin_return)[0][0]+1)/len(sector_rating_all)

        # issuer-wise percentile
        issuer = returns.Ticker[returns.ISIN==isin].values[0]
        issuer_all = returns.loc[returns.Ticker==issuer].one_week_returns.values
        issuer_all = np.array(sorted(issuer_all))
        issuer_perc = (np.where(issuer_all==isin_return)[0][0]+1)/len(issuer_all)

    return sector_rating_perc, issuer_perc


# ### Test for one day
# - 2018-04-10 to 2018-04-16

# In[34]:


get_ipython().run_cell_magic('time', '', "\ndatestr = '2018-04-12'\n\n# find start and end dates\nsorted_dates = curve.date.drop_duplicates().reset_index(drop=True).values\nind1 = np.where(sorted_dates==datestr)[0][0]\nind2 = np.where(sorted_dates==datestr)[0][0]\ntest_period_dates = sorted_dates[ind1:(ind2+1)]\n\n# save final outputs include rich/cheap ISINs, ranking scores, performance percentile\ncolnames = ['ISIN', 'spd', 'iss', 'sec', 'ranking', 'BCLASS3', 'Ticker', 'sector_percentile','issuer_percentile']\nrich_recomm = pd.DataFrame(columns=colnames)\ncheap_recomm = pd.DataFrame(columns=colnames)\n\n# start backtesting\nfor start_date in test_period_dates:\n    \n    print(start_date)\n    \n    # get rich and cheap bonds for this day\n    richb, cheapb = rich_cheap(start_date, curve)\n    \n    # get all bond one-week-returns if buy on this day\n    returns = weekly_returns_on_a_day(start_date, curve)\n    \n    # add two columns to save percentiles\n    richb['sector_percentile'] = -1\n    richb['issuer_percentile'] = -1\n    cheapb['sector_percentile'] = -1\n    cheapb['issuer_percentile'] = -1\n    \n    # for each bond, its return level among the same sector_rating or same issuer (percentile)\n    rich_sector_percentile = []\n    rich_issuer_percentile = []\n    for row, isin in enumerate(richb.ISIN):\n        sec_perc, iss_perc = get_percentile(isin, returns)\n        rich_sector_percentile.append(sec_perc)\n        rich_issuer_percentile.append(iss_perc)\n    richb['sector_percentile'] = rich_sector_percentile\n    richb['issuer_percentile'] = rich_issuer_percentile\n    \n    cheap_sector_percentile = []\n    cheap_issuer_percentile = []\n    for row, isin in enumerate(cheapb.ISIN):\n        sec_perc, iss_perc = get_percentile(isin, returns)\n        cheap_sector_percentile.append(sec_perc)\n        cheap_issuer_percentile.append(iss_perc)\n    cheapb['sector_percentile'] = cheap_sector_percentile\n    cheapb['issuer_percentile'] = cheap_issuer_percentile \n    \n    # append this one day's results to master dataframe\n    rich_recomm = rich_recomm.append(richb, ignore_index=True)\n    cheap_recomm = cheap_recomm.append(cheapb, ignore_index=True)")


# In[35]:


rich_recomm.to_csv('{}_bt_rich.csv'.format(datestr))
cheap_recomm.to_csv('{}_bt_cheap.csv'.format(datestr))


# ## Loop Backtest

# In[52]:


dates = ['2018-04-25', '2018-04-26', '2018-04-27', '2018-04-30']


# In[53]:


for datestr in dates:
     # find start and end dates
    sorted_dates = curve.date.drop_duplicates().reset_index(drop=True).values
    ind1 = np.where(sorted_dates==datestr)[0][0]
    ind2 = np.where(sorted_dates==datestr)[0][0]
    test_period_dates = sorted_dates[ind1:(ind2+1)]

    # save final outputs include rich/cheap ISINs, ranking scores, performance percentile
    colnames = ['ISIN', 'spd', 'iss', 'sec', 'ranking', 'BCLASS3', 'Ticker', 'sector_percentile','issuer_percentile']
    rich_recomm = pd.DataFrame(columns=colnames)
    cheap_recomm = pd.DataFrame(columns=colnames)

    # start backtesting
    for start_date in test_period_dates:

        print(start_date)

        # get rich and cheap bonds for this day
        richb, cheapb = rich_cheap(start_date, curve)

        # get all bond one-week-returns if buy on this day
        returns = weekly_returns_on_a_day(start_date, curve)

        # add two columns to save percentiles
        richb['sector_percentile'] = -1
        richb['issuer_percentile'] = -1
        cheapb['sector_percentile'] = -1
        cheapb['issuer_percentile'] = -1

        # for each bond, its return level among the same sector_rating or same issuer (percentile)
        rich_sector_percentile = []
        rich_issuer_percentile = []
        for row, isin in enumerate(richb.ISIN):
            sec_perc, iss_perc = get_percentile(isin, returns)
            rich_sector_percentile.append(sec_perc)
            rich_issuer_percentile.append(iss_perc)
        richb['sector_percentile'] = rich_sector_percentile
        richb['issuer_percentile'] = rich_issuer_percentile

        cheap_sector_percentile = []
        cheap_issuer_percentile = []
        for row, isin in enumerate(cheapb.ISIN):
            sec_perc, iss_perc = get_percentile(isin, returns)
            cheap_sector_percentile.append(sec_perc)
            cheap_issuer_percentile.append(iss_perc)
        cheapb['sector_percentile'] = cheap_sector_percentile
        cheapb['issuer_percentile'] = cheap_issuer_percentile 

        # append this one day's results to master dataframe
        rich_recomm = rich_recomm.append(richb, ignore_index=True)
        cheap_recomm = cheap_recomm.append(cheapb, ignore_index=True)

    rich_recomm.to_csv('{}_bt_rich.csv'.format(datestr))
    cheap_recomm.to_csv('{}_bt_cheap.csv'.format(datestr))


# #### Analyze results..

# In[137]:


dates = ['2018-04-11', '2018-04-12', '2018-04-13', '2018-04-16', '2018-04-17', 
         '2018-04-18','2018-04-19', '2018-04-20','2018-04-23']


# In[138]:


rich = pd.read_csv('2018-04-10_bt_rich.csv')
cheap = pd.read_csv('2018-04-10_bt_cheap.csv')
rich.shape


# In[139]:


for datestr in dates:
    f = pd.read_csv('{}_bt_rich.csv'.format(datestr))
    rich = rich.append(f)
    
    f1 = pd.read_csv('{}_bt_cheap.csv'.format(datestr))
    cheap = cheap.append(f1)


# In[140]:


# prediction by sector
#r = rich_recomm
r = rich
r_bysec = r[['BCLASS3','sector_percentile']] 

fig, ax = plt.subplots(figsize=(16,5))
plt.suptitle('')
r_bysec.boxplot(by='BCLASS3', ax=ax)
plt.title('Rank of One-Week Returns of Predicted Rich Bonds', fontsize=16)
plt.xlabel('Sectors', fontsize=16)
plt.xticks(rotation = 45)
plt.show()


# In[141]:


# prediction by issuer
r_byiss = r.groupby(by='Ticker').median().reset_index()[['Ticker','issuer_percentile']]
r_byiss = r_byiss.sort_values(by='issuer_percentile').loc[r_byiss.issuer_percentile>0]

plt.figure(figsize=(12,6))
plt.plot(list(range(r_byiss.shape[0])), r_byiss.issuer_percentile)
plt.axhline(y=0.5, linestyle='--', color='grey')
plt.title('Mean Performance of Predicted Rich Bonds by Issuer', fontsize=14)
plt.xlabel('Individual Issuers: %d issuers in total'%r_byiss.shape[0], fontsize=14)
plt.show()


# In[142]:


# prediction by sector
#c = cheap_recomm
c = cheap
c_bysec = c[['BCLASS3','sector_percentile']] 

fig, ax = plt.subplots(figsize=(16,5))
plt.suptitle('')
c_bysec.boxplot(by='BCLASS3', ax=ax)
plt.title('Rank of One-Week Returns of Predicted Cheap Bonds', fontsize=16)
plt.xlabel('Sectors', fontsize=16)
plt.xticks(rotation = 45)
plt.show()


# In[143]:


# prediction by issuer
c_byiss = c.groupby(by='Ticker').median().reset_index()[['Ticker','issuer_percentile']]
c_byiss = c_byiss.sort_values(by='issuer_percentile').loc[c_byiss.issuer_percentile>0]

plt.figure(figsize=(12,6))
plt.plot(list(range(c_byiss.shape[0])), c_byiss.issuer_percentile)
plt.axhline(y=0.5, linestyle='--', color='grey')
plt.title('Mean Performance of Predicted Cheap Bonds by Issuer', fontsize=14)
plt.xlabel('Individual Issuers: %d issuers in total'%c_byiss.shape[0], fontsize=14)
plt.show()


# In[129]:


"""
### save outputs

# save G-Spread outputs
gspd = pd.concat([rich_spd, cheap_spd], axis=1)
gspd.columns=['rich', 'cheap']
gspd.to_csv('elin_spd.csv')

# save Issuer Curve outputs
iss = pd.concat([rich_iss, cheap_iss], axis=1)
iss.columns=['rich', 'cheap']
iss.to_csv('elin_issuerCurve.csv')

# save Sector Curve outputs
sec = pd.concat([rich_sec, cheap_sec], axis=1)
sec.columns=['rich', 'cheap']
sec.to_csv('elin_sectorCurve.csv')
"""

