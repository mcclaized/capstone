import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


# -------------------------- Mixed Effect Model ----------------------------

def calculate_premium_mlm(df, group_by):
    md = smf.mixedlm("gspd ~ short_factor + long_factor", df,
                     groups=df[group_by], re_formula="~ short_factor + long_factor")
    mdf = md.fit()

    premium = df.gspd - mdf.fittedvalues

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
        cheap = info[(info.range_ind == 1) & (info.cheap_ind == 1)][
            ['ISIN', 'percentile', 'cheap_ind', 'ticker', 'sector', 'sector_rating']]
        rich = info[(info.range_ind == 1) & (info.rich_ind == 1)][
            ['ISIN', 'percentile', 'rich_ind', 'ticker', 'sector', 'sector_rating']]
    else:
        cheap = info[(info.range_ind == 1) & (info.cheap_ind == 1)][
            ['ISIN', 'percentile', 'cheap_ind']]
        rich = info[(info.range_ind == 1) & (info.rich_ind == 1)][
            ['ISIN', 'percentile', 'rich_ind']]

    return cheap, rich


def predict_mlm(df, today, lookback):
    # extract today's data
    cr = df[df.date == today]

    # extract data of last 30 days
    dates = np.sort(np.unique(df.date))
    today_ind = np.where(dates == today)[0][0]
    yesterday = dates[today_ind - 1]
    last30_dates = dates[(today_ind - lookback):(today_ind)]
    last30 = df[df.date.isin(last30_dates)]

    # reset columns indices
    cr.reset_index(drop=True, inplace=True)
    last30.reset_index(drop=True, inplace=True)

    # calculate short-term factor and long-term factor for each ISIN in last30
    yrtm = last30.yrtm
    last30['short_factor'] = (1 - np.exp(-.1 * yrtm)) / yrtm
    last30['long_factor'] = (1 - np.exp(-.025 * yrtm)) / yrtm

    # add grouping columns
    last30['sector_rating'] = last30.sector + last30.rating.astype(str)
    last30['ticker_date'] = last30.ticker + last30.date.astype(str)
    last30['rating_date'] = last30.rating.astype(str) + last30.date.astype(str)

    # calculate premiums based on spread, and evaluate the first criterion
    premiums_spd = last30[['date', 'ISIN', 'gspd', 'ticker', 'sector', 'rating', 'sector_rating']]
    premiums_spd.columns = ['date', 'ISIN', 'premium',
                            'ticker', 'sector', 'rating', 'sector_rating']

    cheap_spd, rich_spd = evaluate_criterion(premiums_spd, yesterday, include_info=True)

    # get the list of tickers and sector_ratings of bonds which satisfy the first criterion
    sectors = (set(cheap_spd.sector) | set(rich_spd.sector))
    tickers = (set(cheap_spd.ticker) | set(rich_spd.ticker))
    sector_ratings = (set(cheap_spd.sector_rating) | set(rich_spd.sector_rating))

    # drop the ticker and sector_rating columns to prepare for merge later
    cheap_spd = cheap_spd[['ISIN', 'percentile', 'cheap_ind']]
    rich_spd = rich_spd[['ISIN', 'percentile', 'rich_ind']]

    # calculate premiums w.r.t. issuer curve, and evaluate the second criterion
    premiums_t = pd.DataFrame(columns=['date', 'ISIN', 'premium'])
    last30_t_f = last30[last30.ticker.isin(tickers)]

    for s in sectors:
        one_s = last30_t_f[last30_t_f.sector == s]
        one_s['premium'] = calculate_premium_mlm(one_s, 'ticker_date')

        premiums_t = premiums_t.append(one_s[['date', 'ISIN', 'premium']])

    cheap_t, rich_t = evaluate_criterion(premiums_t, yesterday)

    # calculate premium w.r.t. sector_rating curve, and evaluate the third criterion
    premiums_sr = pd.DataFrame(columns=['date', 'ISIN', 'premium'])
    last30_sr_f = last30[last30.sector_rating.isin(sector_ratings)]

    for s in sectors:
        one_s = last30_sr_f[last30_sr_f.sector == s]
        one_s['premium'] = calculate_premium_mlm(one_s, 'rating_date')

        premiums_sr = premiums_sr.append(one_s[['date', 'ISIN', 'premium']])

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
df = pd.read_csv('vanguard_merge.csv', thousands=',')

# extract useful info
df = df[['date', 'ISIN', 'Ticker', 'BCLASS3', 'S&P Rating Num', 'G Spd', 'Years to Mat']]

# rename columns
df.columns = ['date', 'ISIN', 'ticker', 'sector', 'rating', 'gspd', 'yrtm']

# drop any rows with missing values
df = df.dropna()

df = df[df.sector.isin(
    ['Natural Gas', 'Capital Goods', 'Banking'])]

dates = df.date.unique()

# -------------------------- Calculate results ----------------------------
start_idx = np.where(dates == '2018-04-10')[0][0]
end_idx = np.where(dates == '2018-04-30')[0][0]

period1_cheap = pd.DataFrame(columns=['ISIN', 'date'])
period1_rich = pd.DataFrame(columns=['ISIN', 'date'])

for d in dates[start_idx:end_idx + 1]:
    cheap, rich = predict_mlm(df, d, 30)

    cheap_name = 'results/cheap_mlm_' + d
    rich_name = 'results/rich_mlm_' + d

    cheap.to_csv(cheap_name)
    rich.to_csv(cheap_name)

    cheap_bonds = pd.DataFrame()
    cheap_bonds['ISIN'] = cheap.ISIN
    cheap_bonds['date'] = d

    rich_bonds = pd.DataFrame()
    rich_bonds['ISIN'] = rich.ISIN
    rich_bonds['date'] = d

    period1_cheap = pd.concat([period1_cheap, cheap_bonds], ignore_index=True)
    period1_rich = pd.concat([period1_rich, rich_bonds], ignore_index=True)
