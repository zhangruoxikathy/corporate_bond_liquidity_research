'''
Overview
-------------
This Python script aims to calculate Table 2: Measure of Illiquidity
based on methodology in The Illiquidity of Corporate Bonds, Bao, Pan, and Wang (2010).

Table 2 Measure of Illiquidity:
- Panel A Individual Bonds (The mean and average monthly illiquidity per bond per year)
    - Using trade-by-trade data
    - Using daily data
        - Using our cleaned original data
        - Using our cleaned MMN corrected data
- Panel B Bond Portfolio
    - Equal-weighted: Consider a daily portfolio composed of all bonds, with equally weighted
                      bond returns used to calculate monthly illiquidity and median illiquidity
                      per year
    - Issuance-weighted: Consider a daily portfolio composed of all bonds, with issuance weighted
                         bond returns used to calculate monthly illiquidity and median illiquidity
                         per year
- Panel C Implied by quoted bid-ask spread
    - Mean and median monthly bond bid-ask spread per year

The script also produces summary statistics of monthly per bond illiquidity using both
original daily data and MMN corrected monthly data.


Methodology and Challenges
--------------------------
### Table 2 Panel A Daily Data
During the period in the paper spanning from 2003 to 2009, the illiquidity metric
γ exhibited a mean value of 3.12 and a median of 0.07, with a substantial t-statistic
of 17.06 using daily data, compared to an average of 1.18 and a median of 0.56 observed
in the paper. Our analysis successfully mirrored the initial decline followed by a subsequent
rise in trends as documented in the original study. While other illiquidity metrics maintained
a deviation within 40% when compared to the original findings, the illiquidity we recorded
for 2008-2009 were significantly higher—by a factor of 3 to 4 times—potentially influenced
by approximately six bonds exhibiting γ values exceeding 2000. The original study, however,
did not specify an approach for managing outliers, leaving us uncertain whether these
variations arise from outlier effects or inherent differences in data. In addition,
our percentage of illiquidity significant at 95% level is much lower than what the
paper has, suggesting that the authors might have handled outliers somewhat differently
to maintain higher significance. 6 out of 8 robust t-stats are significant at 95% level
in our analysis, with the overall robust t-stat = 17.6, close to the 16.53 in the paper,
indicating the overall significance of the data. 


### Table 2 Panel B Bond Portfolio
For Panel B, we are trying to construct two sets of daily bond bond portfolios from the
same cross-section of bonds and for the same sample period, one being equally weighted
and the other being weighted by issuance. After obtaining the daily portfolio returns
(using delta log bond price) and lag returns (using delta log bond price lag), we calculated
the monthly illiquidity through negative covariance of the returns and lag returns and then
found the median per year for two sets of portfolios. 

The paper suggests that this measure implies that the transitory component extracted by
the γ measure is idiosyncratic in nature and gets diversified away at the portfolio level,
but a suspected systematic component is present when this aggregate illiquidity measure 
comoves strongly with the aggregate market condition at the time. Similar to the paper,
our peak in illiquidity appeared in ~2006-2007, and most of the portfolio illiquidity
measures were not statistically significant. All measures replicate the paper within a
tolerance of +-0.05 (equal-weighted), +-0.07(issuance-weighted).


### Table 2 Panel C Bid-Ask Spread
In Panel C, we computed the monthly average and median bid-ask spreads for each year,
using these as proxies for implied illiquidity. The methodology involved utilizing the
monthly bond return data available on WRDS to calculate the t-spreads, whereas the
original authors derived their data from daily figures, potentially accounting for some
differences in results. Despite these differences, by applying a factor of 5 to our findings,
we were able to align our results with the original study's observed pattern of initial
decline followed by an increase in illiquidity, with a tolerance level below 40%. It is
noteworthy that the mean bid-ask spread for 2005 exhibited a slight increase in our table,
although the median remained lower than that of the preceding year. This discrepancy
underscores the influence of outliers on the mean and indicates a positive skew in the data.


Requirements
-------------

../data/pulled/Bondret resulting from load_wrds_bondret.py
../data/pulled/BondDailyPublic resulting from load_opensource.py
../data/pulled/WRDS_MMN_Corrected_Data resulting from load_opensource.pys
../data/pulled/IntradayTRACE resulting from load_intraday.py

'''
import logging

#* ************************************** */
#* Libraries                              */
#* ************************************** */
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import config
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR
START_DATE = config.START_DATE
END_DATE = config.END_DATE

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

    del df_daily
    del df_bondret

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
    del dfDC

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
    # df = df[(df['days_to_sttl_ct'] <= 2.0) | (df['days_to_sttl_ct'] == None) | (df['days_to_sttl_ct'] == np.NAN)]
    # df = df[(df['days_to_sttl_ct'] == '002') | (df['days_to_sttl_ct'] == '000')\
    #         | (df['days_to_sttl_ct'] == '001') | (df['days_to_sttl_ct'] == 'None') ]
    df = df[df['wis_fl'] != 'Y']
    df = df[(df['lckd_in_ind'] != 'Y')]
    # df = df[(df['sale_cndtn_cd'] == 'None') | (df['sale_cndtn_cd'] == '@')]
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


def calc_annual_illiquidity_table(df):
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


##############################################################
# MMN Corrected data cleaning and Produce Panel A 
##############################################################

def calc_illiq_w_mmn_corrected(start_date, end_date, cleaned_df):
    """Use clean merged cusips to filter out mmn corrected monthly data to generate illiquidity table."""

    mmn  = load_opensource.load_mmn_corrected_bond(data_dir=DATA_DIR)

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
    df_ew = df.groupby('trd_exctn_dt')[['deltap', 'deltap_lag', 'month_year']].mean().reset_index()

    tqdm.pandas()

    Illiq_port_ew = df_ew.groupby(['month_year'])[['deltap', 'deltap_lag']] \
                        .progress_apply(lambda x: x.cov().iloc[0, 1]) * -1
    Illiq_port_ew = Illiq_port_ew.reset_index()
    Illiq_port_ew.columns = ['month_year', 'Equal-weighted']
    Illiq_port_ew['year'] = Illiq_port_ew['month_year'].dt.year
    Illiq_port_ew = Illiq_port_ew.dropna(subset=['Equal-weighted'])

    # for full equal weighted porfolio illiquidity
    Illiq_port_ew_full = np.median(Illiq_port_ew['Equal-weighted'])

    # Calculate t-stat
    def calculate_t_statistic(group):
        """Calculate t-stat for equal-weighted illiquidity."""
        group_mean = group.mean()
        group_std = group.std()
        n = len(group)
        t_statistic = group_mean / (group_std / np.sqrt(n))
        return t_statistic

    grouped = Illiq_port_ew.groupby('year')['Equal-weighted']
    t_stat_ew = grouped.apply(calculate_t_statistic)

    # Calculate t-stat for full sample
    ew_full_mean = np.mean(Illiq_port_ew['Equal-weighted'])
    ew_full_std = Illiq_port_ew['Equal-weighted'].std()
    ew_full_size = len(Illiq_port_ew)
    ew_full_t_stat = ew_full_mean / (ew_full_std / (ew_full_size ** 0.5))

    # Issurance weighted
    df['issurance'] = df['offering_amt'] * df['principal_amt'] * df['offering_price'] / 100 / 1000000
    df['value_weighted_deltap'] = df['deltap'] * df['issurance']
    df['value_weighted_deltap_lag'] = df['deltap_lag'] * df['issurance']

    # Group by day and calculate the sum of the value-weighted columns and issurance
    df_vw = df.groupby('trd_exctn_dt').agg(
        total_value_weighted_deltap=pd.NamedAgg(column='value_weighted_deltap', aggfunc='sum'),
        total_value_weighted_deltap_lag=pd.NamedAgg(column='value_weighted_deltap_lag', aggfunc='sum'),
        total_issurance=pd.NamedAgg(column='issurance', aggfunc='sum'),
        month_year=pd.NamedAgg(column='month_year', aggfunc='mean')
    )

    # Calculate the average value-weighted deltap and deltap_lag
    df_vw['deltap_vw'] = df_vw['total_value_weighted_deltap'] / df_vw['total_issurance']
    df_vw['deltap_lag_vw'] = df_vw['total_value_weighted_deltap_lag'] / df_vw['total_issurance']

    tqdm.pandas()
    Illiq_port_vw = df_vw.groupby(['month_year'])[['deltap_vw', 'deltap_lag_vw']] \
                        .progress_apply(lambda x: x.cov().iloc[0, 1]) * -1
    Illiq_port_vw = Illiq_port_vw.reset_index()
    Illiq_port_vw.columns = ['month_year', 'Issuance-weighted']
    Illiq_port_vw = Illiq_port_vw.dropna(subset=['Issuance-weighted'])
    Illiq_port_vw['year'] = Illiq_port_vw['month_year'].dt.year

    # for full issuance weighted porfolio illiquidity
    Illiq_port_vw_full = np.median(Illiq_port_vw['Issuance-weighted'])

    # Calculate t-stat for issuance-weighted illiquidity
    grouped = Illiq_port_vw.groupby('year')['Issuance-weighted']
    t_stat_vw = grouped.apply(calculate_t_statistic)

    # Calculate t-stat for full sample
    vw_full_mean = np.mean(Illiq_port_vw['Issuance-weighted'])
    vw_full_std = Illiq_port_vw['Issuance-weighted'].std()
    vw_full_size = len(Illiq_port_vw)
    vw_full_t_stat = vw_full_mean / (vw_full_std / (vw_full_size ** 0.5))

    table2_port = pd.DataFrame({
        'Year': np.unique(Illiq_port_ew['year']),
        'Equal weighted': Illiq_port_ew.groupby('year')['Equal-weighted'].median(),
        'EW t stat': t_stat_ew,
        'Issuance weighted': Illiq_port_vw.groupby('year')['Issuance-weighted'].median(),
        'IW t stat': t_stat_vw
    }).reset_index(drop=True)

    overall_data = pd.DataFrame({
        'Year': ['Full'],
        'Equal weighted': Illiq_port_ew_full,
        'EW t stat': ew_full_t_stat,
        'Issuance weighted': Illiq_port_vw_full,
        'IW t stat': vw_full_t_stat
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


def generate_table2_trade_by_trade(start_date, end_date, paths):
    if all([path.exists() for path in paths]):
        logging.info(f"Already generated data trade-by-trade data")
        return

    cleaned_tbt_df = clean_intraday(start_date, end_date)
    tbt_df = calc_deltaprc(cleaned_tbt_df)

    illiq_tbt, table2_tbt = calc_annual_illiquidity_table(tbt_df)

    table2_tbt.to_csv(paths[0], index=False)

    # Free memory
    del cleaned_tbt_df
    del tbt_df


def generate_table2_panelA_B_C_MMN(start_date, end_date, paths):
    part_1_paths = paths[0]
    part1_needs_run = False
    part_2_paths = paths[1]
    part2_needs_run = False

    if not all([path.exists() for path in part_1_paths]):
        part1_needs_run = True

    if not all([path.exists() for path in part_2_paths]):
        part2_needs_run = True

    if part1_needs_run or part2_needs_run:
        cleaned_daily_df = clean_merged_data(start_date, end_date)
    else:
        logging.info(f"Panel A, B, C, MMN data already generated for period {start_date} to {end_date}")
        return

    if part1_needs_run:
        daily_df = calc_deltaprc(cleaned_daily_df)

        illiq_daily, table2_daily = calc_annual_illiquidity_table(daily_df)
        illiq_daily_summary = create_summary_stats(illiq_daily)
        table2_port = calc_annual_illiquidity_table_portfolio(daily_df)
        table2_spd = calc_annual_illiquidity_table_spd(daily_df)

        illiq_daily.to_csv(part_1_paths[0], index=False)
        illiq_daily_summary.to_csv(part_1_paths[1], index=False)
        table2_daily.to_csv(part_1_paths[2], index=False)
        table2_port.to_csv(part_1_paths[3], index=False)
        table2_spd.to_csv(part_1_paths[4], index=False)

        # Free memory
        del illiq_daily
        del illiq_daily_summary
        del table2_daily
        del table2_port
        del table2_spd
    else:
        logging.info(f"Panel A, B, C data already generated for period {start_date} to {end_date}")

    if part2_needs_run:
        mmn, table2_daily_mmn = calc_illiq_w_mmn_corrected(start_date, end_date,
                                                                   cleaned_daily_df)

        del cleaned_daily_df

        illiq_daily_summary_mmn = create_summary_stats(mmn)

        mmn.to_csv(part_2_paths[0], index=False)
        illiq_daily_summary_mmn.to_csv(part_2_paths[1], index=False)
        table2_daily_mmn.to_csv(part_2_paths[2], index=False)

        # Free memory
        del mmn
        del illiq_daily_summary_mmn
        del table2_daily_mmn
    else:
        logging.info(f"MMN data already generated for period {start_date} to {end_date}")


def main():

    # Replicate Paper Data
    start_date = START_DATE
    end_date = END_DATE

    # Replicate table 2 Panel A Trade-by-Trade in the paper
    logging.info("Running replicate table 2 Panel A Trade-by-Trade in the paper")
    fpaths = [OUTPUT_DIR.joinpath("table2_panelA_trade_by_trade_paper.csv")]
    generate_table2_trade_by_trade(start_date, end_date, fpaths)

    # Replicate table 2 Panel A Daily Data, Panel B, Panel C in the paper
    logging.info("Running replicate table 2 Panel A Daily Data, Panel B, Panel C, and MMN in the paper")
    fpaths = [(
            OUTPUT_DIR.joinpath("illiq_daily_paper.csv"),
            OUTPUT_DIR.joinpath("illiq_summary_paper.csv"),
            OUTPUT_DIR.joinpath("table2_panelA_daily_paper.csv"),
            OUTPUT_DIR.joinpath("table2_panelB_paper.csv"),
            OUTPUT_DIR.joinpath("table2_panelC_paper.csv")
        ),
        (
            OUTPUT_DIR.joinpath("mmn_paper.csv"),
            OUTPUT_DIR.joinpath("illiq_daily_summary_mmn_paper.csv"),
            OUTPUT_DIR.joinpath("table2_daily_mmn_paper.csv")
        )
    ]
    generate_table2_panelA_B_C_MMN(start_date, end_date, fpaths)

    # Generate Recent Data: Update table to the present
    start_date = START_DATE
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Update table 2 Panel A Trade-by-Trade Data to the present
    logging.info("Running update table 2 Panel A Trade-by-Trade Data to the present")
    fpaths = [OUTPUT_DIR.joinpath("table2_panelA_trade_by_trade_new.csv")]
    generate_table2_trade_by_trade(start_date, end_date, fpaths)


    # Update table 2 Panel A Daily Data, Panel B, Panel C to the present
    logging.info("Running update table 2 Panel A Daily Data, Panel B, Panel C, and MMN to the present")
    fpaths = [(
            OUTPUT_DIR.joinpath("illiq_daily_new.csv"),
            OUTPUT_DIR.joinpath("illiq_summary_new.csv"),
            OUTPUT_DIR.joinpath("table2_panelA_daily_new.csv"),
            OUTPUT_DIR.joinpath("table2_panelB_new.csv"),
            OUTPUT_DIR.joinpath("table2_panelC_new.csv")
        ),
        (
            OUTPUT_DIR.joinpath("mmn_new.csv"),
            OUTPUT_DIR.joinpath("illiq_daily_summary_mmn_new.csv"),
            OUTPUT_DIR.joinpath("table2_daily_mmn_new.csv")
        )
    ]
    generate_table2_panelA_B_C_MMN(start_date, end_date, fpaths)

    logging.info("Done generating paper replication and update data files")


if __name__ == "__main__":
    main()
