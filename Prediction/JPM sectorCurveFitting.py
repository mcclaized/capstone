#!/usr/bin/env python
# coding: utf-8
# Yilin Sun

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# read data
df = pd.read_csv('vanguard_merge.csv')

# remove blanks from column names
df.columns = df.columns.str.strip().str.replace('-', '').str.replace(' ','')

# keep only relevant columns
curve = df[['date', 'BCLASS3', 'ISIN', 'OAS', 'PxClose','YearstoMat', 'Ticker']]

# calculate premium
curve['Premium'] = list(curve.PxClose - 100)

# transform Year to Maturity using exponential transformation
# to prepare for curve fitting
ytm = curve.YearstoMat.values
curve['yr5'] = (1 - np.exp(-.2 * ytm))/ytm
curve['yr10'] = (1 - np.exp(-.1* ytm))/ytm


# #### Find Sector Curve


# Sector Curve
def sectorCurve(input_date, curve):

    # sort values - not necessary but suggested
    curve = curve.sort_values(by=['date', 'BCLASS3', 'ISIN'])

    # initialize
    distance = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])
    sectors = np.unique(curve.BCLASS3)
    dates = np.unique(curve.date)

    # index location of the query day
    today_ind = np.where(dates == input_date)[0][0]

    # calculate distance matrix for all bonds all 30 days
    for day in dates[(today_ind-30):(today_ind+1)]:
        oneday = curve[curve.date == day]
        print(day)

        # save all sector's distance in one dataframe
        onedayDist = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])

        for sector in sectors:
            onesector = oneday[oneday.BCLASS3 == sector]
            #print('Sector is ', sector)

            # get y
            y = onesector.OAS.values

            # check if this sector has more than 5 bonds/day
            if len(y) > 5:

                # get x
                x = np.array((onesector.yr5, onesector.yr10)).T

                # fit linear regresion on OAS, 5yr, 10yr
                lr = linear_model.LinearRegression().fit(x, y)

                # prediction
                ypred = lr.predict(x)

                # distances for each bond at one day
                dist = ypred - y

                # append the distance to output dataframe
                onesector['dist'] = dist
                onedayDist = onedayDist.append(onesector[['date', 'ISIN', 'dist']])

        # after calculating dist for all sectors of one day, append to final dist dataframe
        distance = distance.append(onedayDist)

    # split into the first 30 days and the query day
    base = distance[distance.date != input_date]
    queryday = distance[distance.date == input_date][['ISIN', 'dist']]

    # save final results
    result = pd.DataFrame()
    result['ISIN'] = base.groupby('ISIN').max()['dist'].index
    result['maxi'] = base.groupby('ISIN').max()['dist'].values
    result['mini'] = base.groupby('ISIN').min()['dist'].values
    result['delta'] = result.maxi - result.mini
    result['upper95'] = result.maxi - result.delta*0.05
    result['bottom5'] = result.mini + result.delta*0.05

    # merge
    result = pd.merge(result, queryday, on = 'ISIN', how = 'left').drop_duplicates()

    # keep issuers with at least 10bp spread
    result = result.query('delta >= 10')

    # find the rich and cheap bonds
    result['rich'] = (result.dist - result.upper95)/abs(result.dist - result.upper95)
    result['cheap'] = (result.bottom5 - result.dist)/abs(result.bottom5 - result.dist)

    # filter out rich and cheap bonds
    richbonds = result[result.rich == 1].ISIN
    cheapbonds = result[result.cheap == 1].ISIN

    return result, richbonds, cheapbonds


# call function to get recommendation - sector curve
input_date = '2019-03-07'
metric_for_each_bond_sector, richbond_sector, cheapbond_sector = sectorCurve(input_date, curve)


# #### Find Issuer Curve


# Issuer Curve
def issuerCurve(input_date, curve):

    # sort values - not necessary but suggested
    curve = curve.sort_values(by=['date', 'Ticker', 'ISIN'])

    # initialize
    distance = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])
    issuers = np.unique(curve.Ticker)
    dates = np.unique(curve.date)

    # index location of the query day
    today_ind = np.where(dates == input_date)[0][0]

    # calculate distance matrix for all bonds all 30 days
    for day in dates[(today_ind-30):(today_ind+1)]:
        oneday = curve[curve.date == day]
        print(day)

        # save all sector's distance in one dataframe
        onedayDist = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])

        for issuer in issuers:
            oneissuer = oneday[oneday.Ticker == issuer]

            # get x, y
            y = oneissuer.OAS.values

            # check if this issuer has >= 6 bonds/day
            if len(y) > 5:

                x = np.array((oneissuer.yr5, oneissuer.yr10)).T

                # fit linear regresion on OAS, 5yr, 10yr
                lr = linear_model.LinearRegression().fit(x, y)

                # prediction
                ypred = lr.predict(x)

                # distances for each bond at one day
                dist = ypred - y

                # append the distance to output dataframe
                oneissuer['dist'] = dist
                onedayDist = onedayDist.append(oneissuer[['date', 'ISIN', 'dist']])

        # after calculating dist for all sectors of one day, append to final dist dataframe
        distance = distance.append(onedayDist)

    # split into the first 30 days and the query day
    base = distance[distance.date != input_date]
    queryday = distance[distance.date == input_date][['ISIN', 'dist']]

    # save final results
    result = pd.DataFrame()
    result['ISIN'] = base.groupby('ISIN').max()['dist'].index
    result['maxi'] = base.groupby('ISIN').max()['dist'].values
    result['mini'] = base.groupby('ISIN').min()['dist'].values
    result['delta'] = result.maxi - result.mini
    result['upper95'] = result.maxi - result.delta*0.05
    result['bottom5'] = result.mini + result.delta*0.05

    # merge
    result = pd.merge(result, queryday, on = 'ISIN', how = 'left').drop_duplicates()

    # keep issuers with at least 10bp spread
    result = result.query('delta >= 10')

    # find the rich and cheap bonds
    result['rich'] = (result.dist - result.upper95)/abs(result.dist - result.upper95)
    result['cheap'] = (result.bottom5 - result.dist)/abs(result.bottom5 - result.dist)

    # filter out rich and cheap bonds
    richbonds = result[result.rich == 1].ISIN
    cheapbonds = result[result.cheap == 1].ISIN

    return result, richbonds, cheapbonds


# Recommendation from issuer curve
input_date = '2019-03-07'
metric_for_each_bond_issuer, richbond_issuer, cheapbond_issuer = issuerCurve(input_date, curve)


# #### Get intersection between Sector curve and Issuer curve results

richbond = pd.merge(richbond_sector, richbond_issuer, on = 'ISIN', how = 'inner')
cheapbond = pd.merge(cheapbond_sector, cheapbond_issuer, on = 'ISIN', how = 'inner')


# #### Save final results to output

# concatenate into one dataframe
save = pd.concat([richbond, cheapbond], axis=1)

# save rich & cheap bodns to .csv output
save.columns=['Rich bonds', 'Cheap bonds']
save.to_csv('rich_cheap_bonds_20190307.csv')
