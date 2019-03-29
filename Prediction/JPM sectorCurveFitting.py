#!/usr/bin/env python
# coding: utf-8

# - Bonds Filtering
# - Yilin Sun
# - Latest version: 20190327, added logrithmic fit results
# - Include results from G-spread Fitlering, Sector Curve Filtering and Issuer Curve filtering.

# ### Read data and preprocess

# - Y: G-spread
# - X: Years to Maturity

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# read data
df = pd.read_csv('vanguard_merge.csv')

# remove blanks from column names
df.columns = df.columns.str.strip().str.replace('-', '').str.replace(' ','').str.replace('&','')

# keep only relevant columns
curve = df[['date', 'BCLASS3', 'ISIN', 'GSpd', 'YearstoMat', 'Ticker', 'SPRatingNum']]

# drop rows with NA
curve = curve.dropna()

# transform Year to Maturity(YTM) using exponential transformation preparing for curve fitting
ytm = curve.YearstoMat.values
curve['yr5'] = (1 - np.exp(-.2 * ytm))/ytm
curve['yr10'] = (1 - np.exp(-.1* ytm))/ytm


# ------------ Step 1: G-spread Filtering ------------
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


# ------------ Step 2: Issuer Curve Filtering ------------
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
    for day in dates[(today_ind-31):(today_ind)]:
        print(day)
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
                dist = ypred - y

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


# ------------ Step 3: Sector Curve Filtering ------------
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
    for day in dates[(today_ind-31):(today_ind)]:
        oneday = curve[curve.date == day]
        print(day)

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
                    dist = ypred - y

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


# ------------ Step 4: Merge Results from 3 Filterings ------------

# specify a query date
input_date = '2019-03-07'

# G-Spread filtering
richbonds1, cheapbonds1 = GSpdFiltering(input_date, curve)

print('G-spread filtering results\n')
print('# richbonds: %d' % len(richbonds1))
print('# cheapbonds: %d' % len(cheapbonds1))

# issuer curve
result_iss, richbonds_iss, cheapbonds_iss = issuerCurve(input_date, curve)

print('\n Issuer Curve filtering results\n')
print('# richbonds: %d' % len(richbonds_iss))
print('# cheapbonds: %d' % len(cheapbonds_iss))

# sector curve
result_sec, richbonds_sec, cheapbonds_sec = sectorCurve(input_date, curve)

print('\n Sector Curve filtering results\n')
print('# richbonds: %d' % len(richbonds_sec))
print('# cheapbonds: %d' % len(cheapbonds_sec))

# save G-Spread outputs
save = pd.concat([richbonds1, cheapbonds1], axis=1)
save.columns=['rich', 'cheap']
save.to_csv('elin_gspd.csv')

# save Issuer Curve outputs
save = pd.concat([richbonds_iss, cheapbonds_iss], axis=1)
save.columns=['rich', 'cheap']
save.to_csv('elin_issuerCurve.csv')

# save Sector Curve outputs
save = pd.concat([richbonds_sec, cheapbonds_sec], axis=1)
save.columns=['rich', 'cheap']
save.to_csv('elin_sectorCurve.csv')

# save fina results
rich = list(richbonds1) + list(richbonds_sec) + list(richbonds_iss)
cheap = list(cheapbonds1) + list(cheapbonds_sec) + list(cheapbonds_iss)

save = pd.concat([pd.DataFrame(pd.unique(rich)), pd.DataFrame(pd.unique(cheap))], axis=1)
save.columns=['rich', 'cheap']
save.to_csv('elin.csv')
