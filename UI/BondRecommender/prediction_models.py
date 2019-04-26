import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from BondRecommender.data_loader import MultipleDayDataLoader
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Functions -------------------------------

def calculate_premium(df):
    y = df.oas.values
    X = df[['short_factor', 'long_factor']]

    lr = LinearRegression().fit(X, y)

    y_pred = lr.predict(X)

    premium = y - y_pred

    return premium


def evaluate_criterion(premiums, date):
    grouped = premiums.groupby('ISIN')

    info = grouped.premium.agg([np.max, np.min])
    info.reset_index(inplace=True)
    info_today = premiums[premiums.date == date]
    info = pd.merge(info_today, info, on='ISIN', how='inner')

    info['range'] = info.amax - info.amin
    info['percentile'] = (info.premium - info.amin) / info.range

    info['range_ind'] = (info.range >= 10)
    info['ind'] = info.range_ind * 1 * \
        ((info.percentile > 0.95) * 2 + (info.percentile < 0.05) * 1)

    return info[['ISIN', 'percentile', 'ind']]


def predict_rc(multiple_day_data, date, isins):

    last30 = multiple_day_data.data
    last30.reset_index(inplace=True)

    if date is None:
        date = max(last30["date"].unique())

    # extract useful info
    last30 = last30[['date', 'ISIN', 'Ticker', 'BCLASS3',
                     'S&P Rating Num', 'G Spd', 'Years to Mat']]

    # rename columns
    last30.columns = ['date', 'ISIN', 'ticker', 'sector', 'rating', 'oas', 'yrtm']

    last30['sector_rating'] = last30.sector + last30.rating.astype(str)

    # get target bonds info
    target_bonds = last30[last30.ISIN.isin(isins)]

    # extract input_date's data
    cr = last30[last30.date == date]

    # get the dates, tickers and sector_ratings
    dates = pd.unique(last30.date)
    tickers = pd.unique(target_bonds.ticker)
    sector_ratings = pd.unique(target_bonds.sector_rating)

    # reset columns indices
    cr.reset_index(drop=True, inplace=True)
    last30.reset_index(drop=True, inplace=True)

    # calculate short-term factor and long-term factor for each ISIN in last30
    yrtm = last30.yrtm
    last30['short_factor'] = (1 - np.exp(-.2 * yrtm)) / yrtm
    last30['long_factor'] = (1 - np.exp(-.1 * yrtm)) / yrtm

    # calculate premiums based on spread
    premiums_spd = last30[['date', 'ISIN', 'oas', 'ticker', 'sector', 'rating']]
    premiums_spd.columns = ['date', 'ISIN', 'premium', 'ticker', 'sector', 'rating']

    # calculate premiums w.r.t. issuer curve
    premiums_t = pd.DataFrame(columns=['date', 'ISIN', 'premium'])

    for d in dates:
        one_d = last30[last30.date == d]

        for t in tickers:
            one_dt = one_d[one_d.ticker == t]

            if one_dt.shape[0] >= 6:
                one_dt['premium'] = calculate_premium(one_dt)

                premiums_t = premiums_t.append(one_dt[['date', 'ISIN', 'premium']])

    # calculate premium w.r.t. sector_rating curve, and evaluate the third criterion
    premiums_sr = pd.DataFrame(columns=['date', 'ISIN', 'premium'])

    for d in dates:
        one_d = last30[last30.date == d]

        for s in sector_ratings:
            one_dsr = one_d[one_d.sector_rating == s]

            if one_dsr.shape[0] >= 6:
                one_dsr['premium'] = calculate_premium(one_dsr)

                premiums_sr = premiums_sr.append(one_dsr[['date', 'ISIN', 'premium']])

    # evaluate the criterions
    bonds_spd = evaluate_criterion(premiums_spd, date)
    bonds_t = evaluate_criterion(premiums_t, date)
    bonds_sr = evaluate_criterion(premiums_sr, date)

    # rename columns to prepare for merge
    bonds_spd.rename(columns={'ind': 'spd_ind', 'percentile': 'spd_percentile'},
                     inplace=True)
    bonds_t.rename(columns={'ind': 't_ind', 'percentile': 't_percentile'},
                   inplace=True)
    bonds_sr.rename(columns={'ind': 'sr_ind', 'percentile': 'sr_percentile'},
                    inplace=True)

    # merge cheap/rich bonds
    bonds = bonds_spd.merge(bonds_t, on='ISIN', how='outer').merge(
        bonds_sr, on='ISIN', how='outer')

    # get the target bond results
    bonds = bonds[bonds.ISIN.isin(isins)]

    bonds['spd_ind'] = bonds['spd_ind'].replace({np.nan: 0})
    bonds['t_ind'] = bonds['t_ind'].replace({np.nan: 0})
    bonds['sr_ind'] = bonds['sr_ind'].replace({np.nan: 0})

    bonds['score'] = bonds.spd_ind

    bonds.reset_index(inplace=True, drop=True)

    for i in range(len(isins)):
        if bonds.score.iat[i] == 1:
            if bonds.t_ind.iat[i] != 1 and bonds.sr_ind.iat[i] != 1:
                bonds.score.iat[i] = 0
        if bonds.score.iat[i] == 2:
            if bonds.t_ind.iat[i] != 2 and bonds.sr_ind.iat[i] != 2:
                bonds.score.iat[i] = 0

    # change score to rich/cheap/neutral
    bonds['score'] = bonds['score'].replace(
        {2: 'cheap', 1: 'rich', 0: 'neither'})

    bonds = bonds[['ISIN', 'score']]

    bonds.rename(columns={'score': 'rich/cheap'}, inplace=True)

    return bonds
