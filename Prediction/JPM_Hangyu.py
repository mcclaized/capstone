import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# ----------------------------- Functions -------------------------------

def calculate_premium(df):
    y = df.oas.values
    X = df[['short_factor', 'long_factor']]

    lr = LinearRegression().fit(X, y)

    y_pred = lr.predict(X)

    premium = y - y_pred

    return premium


def evaluate_criterion(premiums, today, include_info=False):
    grouped = premiums.groupby('ISIN')

    info = grouped.premium.agg([np.max, np.min])
    info.reset_index(inplace=True)
    info_today = premiums[premiums.date == today]
    info = pd.merge(info_today, info, on='ISIN', how='inner')

    info['range'] = info.amax - info.amin
    info['percentile'] = (info.premium - info.amin) / info.range

    info['range_ind'] = (info.range >= 10)
    info['cheap_ind'] = (info.percentile > 0.95)
    info['rich_ind'] = (info.percentile < 0.05)

    if include_info:
        info['sector_rating'] = info.sector + info.rating.astype(str)
        cheap = info[(info.range_ind == 1) & (info.cheap_ind == 1)][
            ['ISIN', 'percentile', 'cheap_ind', 'ticker', 'sector_rating']]
        rich = info[(info.range_ind == 1) & (info.rich_ind == 1)][
            ['ISIN', 'percentile', 'rich_ind', 'ticker', 'sector_rating']]
    else:
        cheap = info[(info.range_ind == 1) & (info.cheap_ind == 1)][
            ['ISIN', 'percentile', 'cheap_ind']]
        rich = info[(info.range_ind == 1) & (info.rich_ind == 1)][
            ['ISIN', 'percentile', 'rich_ind']]

    return cheap, rich


def predict(alldata, today, lookback):
    # extract today's data
    cr = all[all.date == today]

    # extract data of last 30 days
    dates = np.sort(np.unique(all.date))
    today_ind = np.where(dates == today)[0][0]
    yesterday = dates[today_ind - 1]
    last30_dates = dates[(today_ind - lookback):(today_ind)]
    last30 = all[all.date.isin(last30_dates)]

    # reset columns indices
    cr.reset_index(drop=True, inplace=True)
    last30.reset_index(drop=True, inplace=True)

    # calculate short-term factor and long-term factor for each ISIN in last30
    yrtm = last30.yrtm
    last30['short_factor'] = (1 - np.exp(-.2 * yrtm)) / yrtm
    last30['long_factor'] = (1 - np.exp(-.1 * yrtm)) / yrtm

    # calculate premiums based on spread, and evaluate the first criterion
    premiums_spd = last30[['date', 'ISIN', 'oas', 'ticker', 'sector', 'rating']]
    premiums_spd.columns = ['date', 'ISIN', 'premium', 'ticker', 'sector', 'rating']

    cheap_spd, rich_spd = evaluate_criterion(premiums_spd, yesterday, include_info=True)

    # get the list of tickers and sector_ratings of bonds which satisfy the first criterion

    tickers = (set(cheap_spd.ticker) | set(rich_spd.ticker))
    sector_ratings = (set(cheap_spd.sector_rating) | set(rich_spd.sector_rating))

    # drop the ticker and sector_rating columns

    cheap_spd = cheap_spd[['ISIN', 'percentile', 'cheap_ind']]
    rich_spd = rich_spd[['ISIN', 'percentile', 'rich_ind']]

    # calculate premiums w.r.t. issuer curve, and evaluate the second criterion
    premiums_t = pd.DataFrame(columns=['date', 'ISIN', 'premium'])

    for d in last30_dates:
        one_d = last30[last30.date == d]

        for t in tickers:
            one_dt = one_d[one_d.ticker == t]

            if one_dt.shape[0] >= 6:
                one_dt['premium'] = calculate_premium(one_dt)

                premiums_t = premiums_t.append(one_dt[['date', 'ISIN', 'premium']])

    cheap_t, rich_t = evaluate_criterion(premiums_t, yesterday)

    # calculate premium w.r.t. sector_rating curve, and evaluate the third criterion
    premiums_sr = pd.DataFrame(columns=['date', 'ISIN', 'premium'])

    for d in last30_dates:
        one_d = last30[last30.date == d]
        one_d['sector_rating'] = one_d.sector + one_d.rating.astype(str)

        for s in sector_ratings:
            one_dsr = one_d[one_d.sector_rating == s]

            if one_dsr.shape[0] >= 6:
                one_dsr['premium'] = calculate_premium(one_dsr)

                premiums_sr = premiums_sr.append(one_dsr[['date', 'ISIN', 'premium']])

    cheap_sr, rich_sr = evaluate_criterion(premiums_sr, yesterday)

    # rename columns to prepare for merge
    cheap_spd.rename(columns={'cheap_ind': 'spd_ind', 'percentile': 'spd_percentile'},
                     inplace=True)
    rich_spd.rename(columns={'rich_ind': 'spd_ind', 'percentile': 'spd_percentile'},
                    inplace=True)
    cheap_t.rename(columns={'cheap_ind': 't_ind', 'percentile': 't_percentile'},
                   inplace=True)
    rich_t.rename(columns={'rich_ind': 't_ind', 'percentile': 't_percentile'},
                  inplace=True)
    cheap_sr.rename(columns={'cheap_ind': 'sr_ind', 'percentile': 'sr_percentile'},
                    inplace=True)
    rich_sr.rename(columns={'rich_ind': 'sr_ind', 'percentile': 'sr_percentile'},
                   inplace=True)

    # merge cheap/rich bonds
    cheap = cheap_spd.merge(cheap_t, on='ISIN', how='left').merge(
        cheap_sr, on='ISIN', how='left')
    rich = rich_spd.merge(rich_t, on='ISIN', how='left').merge(
        rich_sr, on='ISIN', how='left')

    # change null values to false in the indicator columns
    cheap[['t_ind', 'sr_ind']] = cheap[['t_ind', 'sr_ind']].fillna(False)
    rich[['t_ind', 'sr_ind']] = rich[['t_ind', 'sr_ind']].fillna(False)

    # calculate the scores to prepare for filtering and ranking
    cheap['score'] = cheap.t_ind * 1 + cheap.sr_ind * 1
    rich['score'] = rich.t_ind * 1 + rich.sr_ind * 1

    # drop the bonds which satisfy neither ticker and sector criterion
    cheap = cheap[cheap.score > 0]
    rich = rich[rich.score > 0]

    # rank bonds
    cheap.sort_values(by=['score', 'spd_percentile'], ascending=False, inplace=True)
    rich.sort_values(by=['score', 'spd_percentile'], ascending=False, inplace=True)

    cheap.reset_index(drop=True, inplace=True)
    rich.reset_index(drop=True, inplace=True)

    return cheap, rich

# ------------------------ Read and clean data -------------------------


all = pd.read_csv('vanguard_merge.csv', thousands=',')

# extract useful info
all = all[['date', 'ISIN', 'Ticker', 'BCLASS3', 'S&P Rating Num', 'G Spd', 'Years to Mat']]

# rename columns
all.columns = ['date', 'ISIN', 'ticker', 'sector', 'rating', 'oas', 'yrtm']

all = all.dropna()

# -------------------------- Calculate results ----------------------------

cheap, rich = predict(all, '2019-03-07', 31)


cheap.shape
rich.shape

# today = '2019-03-07'
# elin = pd.read_csv('elin_gspd.csv', index_col=0)
#
# elin
#
# len(set(cheap_spd.ISIN) & set(elin['cheap']))
# len(set(rich_spd.ISIN) & set(elin['rich']))
# list = [e for e in cheap_spd.ISIN if (e not in set(elin['cheap']))]
# list
