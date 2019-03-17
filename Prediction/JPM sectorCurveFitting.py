#!/usr/bin/env python
# coding: utf-8
# Yilin Sun

# sectorCurve fitting
# data file:  merged .read_csv

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# read data
df = pd.read_csv('vanguard_merge.csv')

# remove blanks from column names
df.columns = df.columns.str.strip().str.replace('-', '').str.replace(' ','')

# subset columns
curve = df[['date', 'BCLASS3', 'ISIN', 'OAS', 'PxClose','YearstoMat']]

# calculate premium
curve['Premium'] = list(curve.PxClose - 100)

# transform Year to Maturity using exponential transformation
# to prepare for curve fitting
ytm = curve.YearstoMat.values
curve['yr5'] = (1 - np.exp(-.2 * ytm))/ytm
curve['yr10'] = (1 - np.exp(-.1* ytm))/ytm

# sort values - not necessary but suggested
curve = curve.sort_values(by=['BCLASS3', 'date', 'ISIN'])

# define function to calculate rich and cheap bonds
def sectorCurve(input_date):
    distance = pd.DataFrame(columns = ['date', 'ISIN', 'dist'])
    sectors = np.unique(curve.BCLASS3)
    dates = np.unique(curve.date)

    # index location of the query day
    today_ind = np.where(dates == input_date)[0][0]

    # calculate distance matrix for all bonds all 30 days
    for day in dates[(today_ind-30):(today_ind+1)]:
        oneday = curve[curve.date == day]
        print(day)

        for sector in sectors:
            onesector = oneday[oneday.BCLASS3 == sector]

            # get x, y
            y = oneday.OAS.values
            x = np.array((oneday.yr5, oneday.yr10)).T

            # fit linear regresion on OAS, 5yr, 10yr
            lr = linear_model.LinearRegression().fit(x, y)

            # prediction
            ypred = lr.predict(x)

            # distances for each bond at one day
            dist = ypred - y

            # append the distance to output dataframe
            oneday['dist'] = dist
            onedaydist = oneday[['date', 'ISIN', 'dist']]
            distance = distance.append(onedaydist)

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

    # find the rich and cheap bonds
    result['rich'] = (result.dist - result.upper95)/abs(result.dist - result.upper95)
    result['cheap'] = (result.bottom5 - result.dist)/abs(result.bottom5 - result.dist)

    # filter out rich and cheap bonds
    richbonds = result[result.rich == 1].ISIN
    cheapbonds = result[result.cheap == 1].ISIN

    return result, richbonds, cheapbonds


# call function to get recommendation
input_date = '2019-03-07'
metric_for_each_bond, richbond, cheapbond = sectorCurve(input_date)

# print rich and cheap bonds
print('Rich bonds: ', richbond)
print('Cheap bonds: ', cheapbond)
