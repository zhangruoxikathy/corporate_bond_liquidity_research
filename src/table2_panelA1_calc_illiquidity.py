'''
Overview
-------------
This Python script aims to calculate the intraday illiquidity based on methodology in
The Illiquidity of Corporate Bonds, Bao, Pan, and Wang (2010).

Requirements
-------------

../data/pulled/IntradayTRACE.parquet resulting from load_intraday.py

'''



import pandas as pd
import numpy as np


def clean_intraday(start_date, end_date):
    df = load_intraday.load_intraday_TRACE()
    df['trd_exctn_dt'] = pd.to_datetime(df['trd_exctn_dt'])
    df['trd_exctn_tm'] = pd.to_datetime(df['trd_exctn_tm'], format='%H:%M:%S').dt.time
    df['trd_tmstamp'] = pd.to_datetime(
        df['trd_exctn_dt'].dt.strftime('%Y-%m-%d') + ' ' + df['trd_exctn_tm'].astype(str))

    # dickerson clean
    df = df[(df['days_to_sttl_ct'] <= 2.0) | (df['days_to_sttl_ct'] == None) | (df['days_to_sttl_ct'] == np.NAN)]
    df = df[df['wis_fl'] != 'Y']
    df = df[(df['lckd_in_ind'] != 'Y')]
    df = df[(df['sale_cndtn_cd'] == 'None') | (df['sale_cndtn_cd'] == '@')]
    df = df[df['entrd_vol_qt'] >= 10000]
    df = df[((df['rptd_pr'] > 5) & (df['rptd_pr'] < 1000))]

    df['month_year'] = df['trd_exctn_dt'].dt.to_period('M')
    df.rename(columns={'rptd_pr': 'prclean', 'cusip_id': 'cusip'}, inplace=True)
    df.sort_values(by=['cusip', 'trd_tmstamp'], inplace=True)
    return df


##############################################################
# Panel A: Individual Bonds, Intraday Data
##############################################################

def calc_deltaprc(df):
    """Calculate delta price and delta price_lag for each intraday trade with additional cleaning.
    """

    # Calculate lagged and lead log prices, and corresponding delta p (percentage returns)
    df['logprc'] = np.log(df['prclean'])
    df['logprc_lag'] = df.groupby('cusip')['logprc'].shift(1)
    df['deltap'] = df['logprc'] - df['logprc_lag']

    # Restrict log returns to be in the interval [1,1]
    df['deltap'] = np.where(df['deltap'] > 1, 1, df['deltap'])
    df['deltap'] = np.where(df['deltap'] < -1, -1, df['deltap'])

    # Convert deltap to % i.e. returns in % as opposed to decimals
    df['deltap'] = df['deltap'] * 100

    # Repeat similar process for deltap_lag
    df['logprc_lead'] = df.groupby('cusip')['logprc'].shift(-1)
    df['deltap_lag'] = df['logprc_lead'] - df['logprc']
    df['deltap_lag'] = np.where(df['deltap_lag'] > 1, 1, df['deltap_lag'])
    df['deltap_lag'] = np.where(df['deltap_lag'] < -1, -1, df['deltap_lag'])
    df['deltap_lag'] = df['deltap_lag'] * 100

    # Drop NAs in deltap, deltap_lag and bonds < 10 observations of the paired price changes
    df_final = df.dropna(subset=['deltap', 'deltap_lag',
                                 'prclean'])  # 'offering_date', 'price_ldm', 'offering_price', 'amount_outstanding'])

    return df_final


def create_annual_illiquidity_table(Illiq_month):
    """Create Panel A illquidity table with cleaned monthly illiquidity data."""

    overall_illiq_mean = np.mean(Illiq_month['illiq'])
    overall_illiq_median = Illiq_month['illiq'].median()

    # Calculate t-statistics for each cusip in each year
    Illiq_month['t stat'] = Illiq_month.groupby(['cusip', 'year'])['illiq'].transform(
        lambda x: (x.mean() / x.sem()) if x.sem() > 0 else np.nan)

    # Identify the entries with t-stat >= 1.96 and calculate the percentage of significant t-stats for each year
    Illiq_month['significant'] = Illiq_month['t stat'] >= 1.96
    percent_significant = Illiq_month.groupby('year')['significant'].mean() * 100
    Illiq_month = Illiq_month.dropna(subset=['illiq', 't stat'])
    overall_percent_significant = Illiq_month['significant'].mean() * 100

    # Calculate robust t-stat for each year
    def get_robust_t_stat(group):
        """Run OLS on a constant term only (mean of illiq) to get the intercept's t-stat."""
        X = add_constant(group['illiq'])
        ols_result = OLS(group['illiq'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

        return abs(ols_result.tvalues[0])

    robust_t_stats = Illiq_month.groupby('year').apply(get_robust_t_stat)

    def calculate_overall_robust_t_stat(series):
        X = add_constant(series)
        ols_result = OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        return abs(ols_result.tvalues[0])

    # Call the function and assign the result to overall_robust_t_stat
    overall_robust_t_stat = calculate_overall_robust_t_stat(Illiq_month['illiq'].dropna())

    # Combine the results
    table2_daily = pd.DataFrame({
        'Year': robust_t_stats.index,
        'Mean illiq': Illiq_month.groupby('year')['illiq'].mean(),
        'Median illiq': Illiq_month.groupby('year')['illiq'].median(),
        'Per t greater 1.96': percent_significant,
        'Robust t stat': robust_t_stats.values
    }).reset_index(drop=True)

    overall_data = pd.DataFrame({
        'Year': ['Full'],
        'Mean illiq': [overall_illiq_mean],
        'Median illiq': [overall_illiq_median],
        'Per t greater 1.96': [overall_percent_significant],
        'Robust t stat': [overall_robust_t_stat]
    })

    table2_daily = pd.concat([table2_daily, overall_data], ignore_index=True)

    return Illiq_month, table2_daily


def calc_annual_illiquidity_table_intraday(df):
    """Calculate illiquidity = -cov(deltap, deltap_lag) using daily data, by month."""

    tqdm.pandas()

    Illiq_month = df.groupby(['cusip', 'month_year'])[['deltap', 'deltap_lag']] \
                      .progress_apply(lambda x: x.cov().iloc[0, 1]) * -1
    Illiq_month = Illiq_month.reset_index()
    Illiq_month.columns = ['cusip', 'month_year', 'illiq']
    Illiq_month['year'] = Illiq_month['month_year'].dt.year
    Illiq_month = Illiq_month.dropna(subset=['illiq'])
    # Illiq_month = Illiq_month[Illiq_month['illiq'] < 2000]  # for outliers
    Illiq_month, table2_daily = create_annual_illiquidity_table(Illiq_month)

    return Illiq_month, table2_daily


def create_summary_stats(df):
    """Calculate relevant summary statistics of the illiquidity data."""

    summary_stats = df.groupby('year').agg({
        'illiq': ['min', 'mean', lambda x: x.quantile(0.25), 'median',
                  lambda x: x.quantile(0.75), 'max', 'std'],
        't stat': 'mean'
    })

    summary_stats.columns = ['min illiq', 'mean illiq', 'q1 0.25', 'median',
                             'q3 0.75', 'max illiq', 'std illiq', 'mean t stat']

    summary_stats.reset_index(inplace=True)

    return summary_stats


def main():
    start_date = '2003-04-14'
    end_date = '2009-06-30'

    # Replicate table 2 in the paper
    cleaned_df_paper = clean_intraday(start_date, end_date)
    df_paper = calc_deltaprc(cleaned_df_paper)

    illiq_intraday_paper, table2_daily_paper = calc_annual_illiquidity_table_intraday(df_paper)
    illiq_daily_summary_paper = create_summary_stats(illiq_intraday_paper)
    table2_port_paper = calc_annual_illiquidity_table_portfolio(df_paper)
    table2_spd_paper = calc_annual_illiquidity_table_spd(df_paper)

    illiq_daily_summary_paper.to_csv(OUTPUT_DIR / "illiq_summary_paper.csv", index=False)
    table2_daily_paper.to_csv(OUTPUT_DIR / "table2_daily_paper.csv", index=False)
    table2_port_paper.to_csv(OUTPUT_DIR / "table2_port_paper.csv", index=False)
    table2_spd_paper.to_csv(OUTPUT_DIR / "table2_spd_paper.csv", index=False)

    # Using MMN corrected data
    mmn_paper, table2_daily_mmn_paper = calc_illiq_w_mmn_corrected(start_date, end_date,
                                                                   cleaned_df_paper)
    illiq_daily_summary_mmn_paper = create_summary_stats(mmn_paper)
    illiq_daily_summary_mmn_paper.to_csv(OUTPUT_DIR / "illiq_daily_summary_mmn_paper.csv", index=False)
    table2_daily_mmn_paper.to_csv(OUTPUT_DIR / "table2_daily_mmn_paper.csv", index=False)

    # Update table to the present
    cleaned_df_new = clean_merged_data('2003-04-14', today)
    df_new = calc_deltaprc(cleaned_df_new)

    illiq_daily_new, table2_daily_new = calc_annual_illiquidity_table_daily(df_new)
    illiq_daily_summary_new = create_summary_stats(illiq_daily_new)
    table2_port_new = calc_annual_illiquidity_table_portfolio(df_new)
    table2_spd_new = calc_annual_illiquidity_table_spd(df_new)

    illiq_daily_summary_new.to_csv(OUTPUT_DIR / "illiq_summary_new.csv", index=False)
    table2_daily_new.to_csv(OUTPUT_DIR / "table2_daily_new.csv", index=False)
    table2_port_new.to_csv(OUTPUT_DIR / "table2_port_new.csv", index=False)
    table2_spd_new.to_csv(OUTPUT_DIR / "table2_spd_new.csv", index=False)

    # Using MMN corrected data
    mmn_new, table2_daily_mmn_new = calc_illiq_w_mmn_corrected(start_date, end_date,
                                                               cleaned_df_new)
    illiq_daily_summary_mmn_new = create_summary_stats(mmn_new)
    illiq_daily_summary_mmn_new.to_csv(OUTPUT_DIR / "illiq_daily_summary_mmn_new.csv", index=False)
    table2_daily_mmn_new.to_csv(OUTPUT_DIR / "table2_daily_mmn_new.csv", index=False)


if __name__ == '__main__':
    main()