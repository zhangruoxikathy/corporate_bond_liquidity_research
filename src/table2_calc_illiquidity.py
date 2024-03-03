'''
Overview
-------------
This Python script aims to calculate illiquidity based on methodology in
The Illiquidity of Corporate Bonds, Bao, Pan, and Wang (2010).
 
Requirements
-------------

../data/pulled/Bondret.parquet resulting from load_wrds_bondret.py
../data/pulled/BondDailyPublic resulting from load_opensource.py
../data/pulled/IntradayTRACE.parquet resulting from load_intraday.py

'''

#* ************************************** */
#* Libraries                              */
#* ************************************** */ 
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
from scipy import stats
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import misc_tools
import load_wrds_bondret
import load_opensource
import load_intraday
import data_processing as data


def clean_merged_data(start_date, end_date):
    """Load merged, pre-cleaned daily and monthly corporate bond data for a given time interval.
    """

    # load and merge pre-cleaned daily and monthly data
    df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)
    df_bondret = load_wrds_bondret.load_bondret(data_dir=DATA_DIR)
    merged_df = data.all_trace_data_merge(df_daily, df_bondret,
                                          start_date = start_date, end_date = end_date)
    merged_df = data.sample_selection(merged_df, start_date = start_date,
                                      end_date = end_date)

    # Clean data
    merged_df = merged_df.dropna(subset=['prclean'])
    merged_df = merged_df.sort_values(by='trd_exctn_dt')
    merged_df['month_year'] = pd.to_datetime(merged_df['trd_exctn_dt']).dt.to_period('M') 

    # Lags days for day_counts
    merged_df['trd_exctn_dt_lag'] = merged_df.groupby('cusip')['trd_exctn_dt'].shift(1)
    dfDC = merged_df.dropna(subset=['trd_exctn_dt_lag'])

    # Generate a list of U.S. holidays over this period
    # Only include "daily" return if the gap between trades is less than 1-Week 
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start_date, end_date)  # 01JUL2002  # 31DEC2022
    holiday_date_list = holidays.date.tolist()

    dfDC['n']  = np.busday_count(dfDC['trd_exctn_dt_lag'].values.astype('M8[D]') , 
                                        dfDC['trd_exctn_dt'].values.astype('M8[D]'),
                                        holidays = holiday_date_list)

    df = merged_df.merge(dfDC[['cusip', 'trd_exctn_dt', 'n']],
                         left_on = ['cusip','trd_exctn_dt'], 
                         right_on = ['cusip','trd_exctn_dt'], how = "left")
    del(dfDC)
    df = df[df.n <= 7]

    return df


def clean_intraday(start_date, end_date):
    """Load pre-cleaned and pre-filtered intraday corporate bond data for a given time interval.
    """
    df = load_intraday.load_intraday_TRACE(start_date, end_date)

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
# Panel A: Individual Bonds, Daily Data
##############################################################


def calc_deltaprc(df):
    """Calculate delta price and delta price_lag for each daily trades with additional cleaning.
    """

    # Calculate lagged and lead log prices, and corresponding delta p (percentage returns)
    df['logprc']     = np.log(df['prclean'])
    df['logprc_lag'] = df.groupby( 'cusip' )['logprc'].shift(1)
    df['deltap']     = df ['logprc'] - df ['logprc_lag']

    # Restrict log returns to be in the interval [1,1]
    df['deltap'] = np.where(df['deltap'] > 1, 1, df['deltap'])
    df['deltap'] = np.where(df['deltap'] <-1, -1, df['deltap'])

    # Convert deltap to % i.e. returns in % as opposed to decimals
    df['deltap'] = df['deltap'] * 100
    
    # Repeat similar process for deltap_lag
    df['logprc_lead'] = df.groupby( 'cusip' )['logprc'].shift(-1)
    df['deltap_lag'] = df ['logprc_lead'] - df ['logprc']
    df['deltap_lag'] = np.where(df['deltap_lag'] > 1, 1, df['deltap_lag'])
    df['deltap_lag'] = np.where(df['deltap_lag'] <-1, -1, df['deltap_lag'])
    df['deltap_lag'] = df['deltap_lag'] * 100

    # Drop NAs in deltap, deltap_lag and bonds < 10 observations of the paired price changes
    df_final = df.dropna(subset=['deltap', 'deltap_lag', 'prclean'])

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
        ols_result = OLS(group['illiq'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})

        return abs(ols_result.tvalues[0])


    robust_t_stats = Illiq_month.groupby('year').apply(get_robust_t_stat)
    
    
    def calculate_overall_robust_t_stat(series):
        X = add_constant(series)
        ols_result = OLS(series, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
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


def calc_annual_illiquidity_table_daily(df):
    """Calculate illiquidity = -cov(deltap, deltap_lag) using daily data, by month."""

    tqdm.pandas()
    
    Illiq_month = df.groupby(['cusip','month_year'] )[['deltap','deltap_lag']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
    Illiq_month = Illiq_month.reset_index()
    Illiq_month.columns = ['cusip','month_year','illiq']
    Illiq_month['year'] = Illiq_month['month_year'].dt.year
    Illiq_month = Illiq_month.dropna(subset=['illiq'])
    # Illiq_month = Illiq_month[Illiq_month['illiq'] < 2000]  # for outliers
    Illiq_month, table2_daily = create_annual_illiquidity_table(Illiq_month)
    
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
    Illiq_month, table2_intraday = create_annual_illiquidity_table(Illiq_month)

    return Illiq_month, table2_intraday




def calc_illiq_w_mmn_corrected(start_date, end_date, cleaned_df):
    """Use clean merged cusips to filter out mmn corrected monthly data to generate illiquidity table."""

    mmn  = load_opensource.load_mmn_corrected_bond(data_dir=DATA_DIR)
    # pd.read_csv('../data/pulled/WRDS_MMN_Corrected_Data.csv.gzip',
    #     compression='gzip')
    
    # Filter out corrected data using cleaned cusips and dates
    mmn = mmn[(mmn['date'] >= start_date) & (mmn['date'] <= end_date)]
    unique_cusip = np.unique(cleaned_df['cusip'])
    mmn = mmn[mmn['cusip'].isin(unique_cusip)]
    
    # Clean data
    mmn['year'] = pd.to_datetime(mmn['date']).dt.to_period('Y') 
    mmn = mmn.dropna(subset=['ILLIQ'])
    mmn['illiq'] = mmn['ILLIQ']
    
    mmn, table2_daily = create_annual_illiquidity_table(mmn)
    
    return mmn, table2_daily


##############################################################
# Summary Statistics Compilation Using Daily Illiquidity Data
##############################################################


def create_summary_stats(illiq_daily):
    """Calculate relevant summary statistics of the illiquidity daily data."""
    
    summary_stats = illiq_daily.groupby('year').agg({
    'illiq': ['min', 'mean', lambda x: x.quantile(0.25), 'median',
              lambda x: x.quantile(0.75), 'max', 'std'],
    't stat': 'mean'
    })
    summary_stats.columns = ['min illiq', 'mean illiq', 'q1 0.25', 'median',
                             'q3 0.75', 'max illiq', 'std illiq', 'mean t stat']
    summary_stats.reset_index(inplace=True)

    return summary_stats


##############################################################
# Panel B: Bond Portfolios
##############################################################

def calc_annual_illiquidity_table_portfolio(df):
    """Calculate illiquidity by using equal weighted and issurance weighted portfolios for each year.
    """
    # Equal weighted
    df_ew = df.groupby('trd_exctn_dt')[['deltap', 'deltap_lag']].mean().reset_index()
    df_ew['year'] = df_ew['trd_exctn_dt'].dt.year

    tqdm.pandas()
    
    Illiq_port_ew = df_ew.groupby(['year'] )[['deltap','deltap_lag']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
    Illiq_port_ew = Illiq_port_ew.reset_index()
    Illiq_port_ew.columns = ['year','Equal-weighted']
    Illiq_port_ew = Illiq_port_ew.dropna(subset=['Equal-weighted'])
    
    # for full equal weighted porfolio illiquidity
    df_ew['full'] = 1 
    Illiq_port_ew_full = df_ew.groupby(['full'] )[['deltap','deltap_lag']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
    
    # Calculate t-stat for equal-weighted illiquidity
    Illiq_port_ew['EW t-stat'] = Illiq_port_ew.apply(
        lambda row: row['Equal-weighted'] / (df_ew[df_ew['year'] == row['year']]['deltap'].std() / 
                                             (len(df_ew[df_ew['year'] == row['year']]) ** 0.5)), axis=1)
    
    # Calculate t-stat for full sample
    ew_full_mean = Illiq_port_ew_full[1]
    ew_full_std = df_ew['deltap'].std()
    ew_full_size = len(df_ew)
    ew_full_t_stat = ew_full_mean / (ew_full_std / (ew_full_size ** 0.5))
    
    # Issurance weighted
    df['issurance'] = df['offering_amt'] * df['principal_amt'] * df['offering_price'] / 100 / 1000000
    df['value_weighted_deltap'] = df['deltap'] * df['issurance']
    df['value_weighted_deltap_lag'] = df['deltap_lag'] * df['issurance']

    # Group by day and calculate the sum of the value-weighted columns and issurance
    df_vw = df.groupby('trd_exctn_dt').agg(
        total_value_weighted_deltap=pd.NamedAgg(column='value_weighted_deltap', aggfunc='sum'),
        total_value_weighted_deltap_lag=pd.NamedAgg(column='value_weighted_deltap_lag', aggfunc='sum'),
        total_issurance=pd.NamedAgg(column='issurance', aggfunc='sum')
    )

    # Calculate the average value-weighted deltap and deltap_lag
    df_vw['deltap_vw'] = df_vw['total_value_weighted_deltap'] / df_vw['total_issurance']
    df_vw['deltap_lag_vw'] = df_vw['total_value_weighted_deltap_lag'] / df_vw['total_issurance']
    df_vw['year'] = df_vw.index.year
    
    tqdm.pandas()
    Illiq_port_vw = df_vw.groupby(['year'])[['deltap_vw','deltap_lag_vw']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
    Illiq_port_vw = Illiq_port_vw.reset_index()
    Illiq_port_vw.columns = ['year','Issuance-weighted']
    Illiq_port_vw = Illiq_port_vw.dropna(subset=['Issuance-weighted'])

    # for full equal weighted porfolio illiquidity
    df_vw['full'] = 1
    Illiq_port_vw_full = df_vw.groupby(['full'] )[['deltap_vw','deltap_lag_vw']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
        
    # Calculate t-stat for issuance-weighted illiquidity
    Illiq_port_vw['IW t-stat'] = Illiq_port_vw.apply(
        lambda row: row['Issuance-weighted'] / (df_vw[df_vw['year'] == row['year']]['deltap_vw'].std() / 
                                                 (len(df_vw[df_vw['year'] == row['year']]) ** 0.5)), axis=1)

    # Calculate t-stat for full sample
    iw_full_mean = Illiq_port_vw_full[1]
    iw_full_std = df_vw['deltap_vw'].std()
    iw_full_size = len(df_vw)
    iw_full_t_stat = iw_full_mean / (iw_full_std / (iw_full_size ** 0.5))

    table2_port = pd.DataFrame({
        'Year': Illiq_port_vw['year'],
        'Equal weighted': Illiq_port_ew['Equal-weighted'],
        'EW t stat': Illiq_port_ew['EW t-stat'],
        'Issuance weighted': Illiq_port_vw['Issuance-weighted'],
        'IW t stat': Illiq_port_vw['IW t-stat']
    }).reset_index(drop=True)
    
    overall_data = pd.DataFrame({
        'Year': ['Full'],
        'Equal weighted': Illiq_port_ew_full,
        'EW t stat': ew_full_t_stat,
        'Issuance weighted': Illiq_port_vw_full,
        'IW t stat': iw_full_t_stat
    })

    table2_port = pd.concat([table2_port, overall_data], ignore_index=True)
    
    return table2_port


##############################################################
# Panel C: Implied Gamma/illiquidity by Quoted Bid-Ask Spreads
##############################################################


def calc_annual_illiquidity_table_spd(df):
    """Calculate mean and median gamma implied by quoted bid-ask spreads by year.
    """
    df_unique = df.groupby(['cusip', 'month_year'])['t_spread'].first().reset_index()
    df_unique['year'] = df_unique['month_year'].dt.year  
    df_unique = df_unique.sort_values(by='month_year')

    Illiq_mean_table = df_unique.groupby('year')['t_spread'].mean()
    overall_illiq_mean = df_unique['t_spread'].mean()
    overall_illiq_median = df_unique['t_spread'].median()
    
    table2_spd = pd.DataFrame({
        'Year': Illiq_mean_table.index,
        'Mean implied gamma': df_unique.groupby('year')['t_spread'].mean(),
        'Median implied gamma': df_unique.groupby('year')['t_spread'].median(),
    }).reset_index(drop=True)
    
    overall_data = pd.DataFrame({
        'Year': ['Full'],
        'Mean implied gamma': [overall_illiq_mean], 
        'Median implied gamma': [overall_illiq_median]
    })
    
    table2_spd = pd.concat([table2_spd, overall_data], ignore_index=True)
    
    return table2_spd



def main():
    
    # Define dates
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = '2003-04-14'
    end_date = '2009-06-30'

    # Replicate table 2 Trade-by-Trade Data in the paper
    cleaned_tbt_df_paper = clean_intraday(start_date, end_date)
    tbt_df_paper = calc_deltaprc(cleaned_tbt_df_paper)

    illiq_tbt_paper, table2_tbt_paper = calc_annual_illiquidity_table_daily(tbt_df_paper)

    table2_tbt_paper.to_csv(OUTPUT_DIR / "table2_tbt_paper.csv", index=False)

    # Free memory
    del cleaned_tbt_df_paper
    del tbt_df_paper
    
    # Replicate table 2 Daily Data in the paper
    cleaned_daily_df_paper = clean_merged_data(start_date, end_date)
    daily_df_paper = calc_deltaprc(cleaned_daily_df_paper)
    
    illiq_daily_paper, table2_daily_paper = calc_annual_illiquidity_table_daily(daily_df_paper)
    illiq_daily_summary_paper = create_summary_stats(illiq_daily_paper)
    table2_port_paper = calc_annual_illiquidity_table_portfolio(daily_df_paper)
    table2_spd_paper = calc_annual_illiquidity_table_spd(daily_df_paper)
    
    
    illiq_daily_summary_paper.to_csv(OUTPUT_DIR / "illiq_summary_paper.csv", index=False)
    table2_daily_paper.to_csv(OUTPUT_DIR / "table2_daily_paper.csv", index=False)
    table2_port_paper.to_csv(OUTPUT_DIR / "table2_port_paper.csv", index=False)
    table2_spd_paper.to_csv(OUTPUT_DIR / "table2_spd_paper.csv", index=False)
    
    # Using MMN corrected data
    mmn_paper, table2_daily_mmn_paper = calc_illiq_w_mmn_corrected(start_date, end_date,
                                                                   cleaned_daily_df_paper)
    illiq_daily_summary_mmn_paper = create_summary_stats(mmn_paper)
    illiq_daily_summary_mmn_paper.to_csv(OUTPUT_DIR / "illiq_daily_summary_mmn_paper.csv", index=False)
    table2_daily_mmn_paper.to_csv(OUTPUT_DIR / "table2_daily_mmn_paper.csv", index=False)

    # Free memory
    del cleaned_daily_df_paper
    del daily_df_paper

    # Update table 2 Trade-by-Trade Data to the present
    cleaned_tbt_df_new = clean_intraday(end_date, today)
    tbt_df_new = calc_deltaprc(cleaned_tbt_df_new)

    illiq_tbt_new, table2_tbt_new = calc_annual_illiquidity_table_daily(tbt_df_new)

    table2_tbt_new.to_csv(OUTPUT_DIR / "table2_tbt_new.csv", index=False)

    # Free memory
    del cleaned_tbt_df_new
    del tbt_df_new

    # Update table 2 Daily Data to the present
    cleaned_daily_df_new = clean_merged_data(end_date, today)
    daily_df_new = calc_deltaprc(cleaned_daily_df_new)

    illiq_daily_new, table2_daily_new = calc_annual_illiquidity_table_daily(daily_df_new)
    illiq_daily_summary_new = create_summary_stats(illiq_daily_new)
    table2_port_new = calc_annual_illiquidity_table_portfolio(daily_df_new)
    table2_spd_new = calc_annual_illiquidity_table_spd(daily_df_new)
    
    illiq_daily_summary_new.to_csv(OUTPUT_DIR / "illiq_summary_new.csv", index=False)
    table2_daily_new.to_csv(OUTPUT_DIR / "table2_daily_new.csv", index=False)
    table2_port_new.to_csv(OUTPUT_DIR / "table2_port_new.csv", index=False)
    table2_spd_new.to_csv(OUTPUT_DIR / "table2_spd_new.csv", index=False)
    
    # Using MMN corrected data
    mmn_new, table2_daily_mmn_new = calc_illiq_w_mmn_corrected(start_date, end_date,
                                                               cleaned_daily_df_new)
    illiq_daily_summary_mmn_new = create_summary_stats(mmn_new)
    illiq_daily_summary_mmn_new.to_csv(OUTPUT_DIR / "illiq_daily_summary_mmn_new.csv", index=False)
    table2_daily_mmn_new.to_csv(OUTPUT_DIR / "table2_daily_mmn_new.csv", index=False)



if __name__ == "__main__":
    main()
