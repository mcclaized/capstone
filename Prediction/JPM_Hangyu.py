import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# Read and clean data


df = pd.read_csv('Chill.csv', thousands=',')
df = df[['Unnamed: 0', 'ISIN', 'Ticker', 'BCLASS3', 'G Spd', 'Years to Mat']]
df.columns = ['date', 'isin', 'issuer', 'sector', 'g_spd', 'yrtm']

for i in range(len(df)):
    if df.at[i, 'date'].startswith('0'):
        df.at[i, 'date'] = df.iat[i, 0][-10:]
    else:
        df.at[i, 'date'] = None


df['date'] = df['date'].ffill()
df = df[pd.notnull(df['isin'])]

# Initialization


today = '02/06/2019'
cr = df[df['date'] == today]
last30 = df
cr.reset_index(drop=True, inplace=True)
last30.reset_index(drop=True, inplace=True)


cols1 = ['issuer_p/d', 'sector_rating_p/d']
cols2 = ['liquidity', 'g', 'issuer', 'sector_rating']

for col in cols1:
    last30[col] = 0.0

for col in cols2:
    cr[col] = 0


# Define functions


def spd_info(col, isin):
    s = last30[last30['isin'] == isin][col]
    max = s.max()
    min = s.min()
    return max-min, min, s.mean()


def mark_col(col_check, col_mark):
    for i in range(len(cr)):
        isin = cr.at[i, 'isin']
        rg, min, mean = spd_info(col_check, isin)

        if rg >= 10:
            if cr.at[i, col_check] >= (min + rg * 0.95):
                cr.at[i, col_mark] = 1
            if cr.at[i, col_check] <= (min + rg * 0.05):
                cr.at[i, col_mark] = -1


# G spread


mark_col('g_spd', 'g')
sum(cr['g'] == 1)
sum(cr['g'] == -1)


# Issuer


lr = LinearRegression()


for issuer in set(last30['issuer']):
    issuer_df = last30[last30['issuer'] == issuer][['date', 'isin', 'g_spd', 'yrtm']]

    for date in set(issuer_df['date']):
        issuer_date_df = issuer_df[issuer_df['date'] == date]
        issuer_date_df['log_yrtm'] = np.log(issuer_date_df['yrtm'])
        lr.fit(issuer_date_df['log_yrtm'].values.reshape(-1, 1), issuer_date_df['g_spd'])

        for i in range(len(issuer_date_df)):
            isin = issuer_date_df['isin'].iat[i]
            log_yrtm = issuer_date_df['log_yrtm'].iat[i]
            g_spd = issuer_date_df['g_spd'].iat[i]
            g_spd_predict = lr.predict(log_yrtm)[0]
            id = last30.index[(df['date'] == date) & (df['isin'] == isin)][0]
            last30.at[id, 'issuer_p/d'] = g_spd - g_spd_predict


cr['issuer_p/d'] = last30[df['date'] == today]['issuer_p/d'].reset_index(drop=True)

mark_col('issuer_p/d', 'issuer')

sum(cr['g'] == 1)
sum(cr['g'] == -1)

sum(cr['issuer'] == 1)
sum(cr['issuer'] == -1)

sum((cr['issuer'] == 1) & (cr['g'] == 1))
sum((cr['issuer'] == -1) & (cr['g'] == -1))
