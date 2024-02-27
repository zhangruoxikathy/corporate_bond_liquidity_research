'''
Overview
-------------
This Python script aims to calculate illiquidity based on methodology in
The Illiquidity of Corporate Bonds, Bao, Pan, and Wang (2010).
 
Requirements
-------------

../data/pulled/Bondret.parquet resulting from load_wrds_bondret.py
../data/pulled/BondDailyPublic resulting from load_opensource.py

'''

#* ************************************** */
#* Libraries                              */
#* ************************************** */ 
import pandas as pd
from tqdm import tqdm
import numpy as np
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


##############################################################
# Panel A: Individual Bonds
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
    
    df['logprc_lead'] = df.groupby( 'cusip' )['logprc'].shift(-1)
    df['deltap_lag'] = df ['logprc_lead'] - df ['logprc']
    df['deltap_lag'] = np.where(df['deltap_lag'] > 1, 1, df['deltap_lag'])
    df['deltap_lag'] = np.where(df['deltap_lag'] <-1, -1, df['deltap_lag'])
    df['deltap_lag'] = df['deltap_lag'] * 100

    # df.isna().sum()
    # df_bondret.columns
    
    # Drop NAs in deltap, deltap_lag and bonds < 10 observations of the paired price changes
    df_final = df.dropna(subset=['deltap', 'deltap_lag', 'prclean'])  # 'offering_date', 'price_ldm', 'offering_price', 'amount_outstanding'])
    df_final['trade_counts'] = df_final.groupby(['cusip', 'year'])['deltap'].transform("count")

    return df_final


def calc_annual_illiquidity_table_daily(df):
    """Calculate illiquidity = -cov(deltap, deltap_lag) using daily data, groupby in month,
    present as annual mean, median, percetage t >= 1.96, and robust t-stat. 
    """
    tqdm.pandas()
    
    Illiq_month = df.groupby(['cusip','month_year'] )[['deltap','deltap_lag']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
    Illiq_month = Illiq_month.reset_index()
    Illiq_month.columns = ['cusip','month_year','illiq']
    Illiq_month['year'] = Illiq_month['month_year'].dt.year
    Illiq_month = Illiq_month.dropna(subset=['illiq'])
    # Illiq_month = Illiq_month[Illiq_month['illiq'] < 2000]

    overall_illiq_mean = np.mean(Illiq_month['illiq'])
    overall_illiq_median = Illiq_month['illiq'].median()

    # Calculate t-statistics for each cusip in each year
    Illiq_month['t_stat'] = Illiq_month.groupby(['cusip', 'year'])['illiq'].transform(
        lambda x: (x.mean() / x.sem()) if x.sem() > 0 else np.nan)

    # Identify the entries with t-stat >= 1.96 and calculate the percentage of significant t-stats for each year
    Illiq_month['significant'] = Illiq_month['t_stat'] >= 1.96
    percent_significant = Illiq_month.groupby('year')['significant'].mean() * 100
    Illiq_month = Illiq_month.dropna(subset=['illiq', 't_stat'])
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

    cleaned_df = clean_merged_data('2003-04-14', '2009-06-30')
    df = calc_deltaprc(cleaned_df)
    # unique_cusip = np.unique(df['cusip'])
    # df_unique_cusip = pd.DataFrame(unique_cusip, columns=['CUSIP'])
    # df_unique_cusip.to_csv("../data/unique_cusips.csv", index=True)
    illiq_daily, table2_daily = calc_annual_illiquidity_table_daily(df)
    table2_spd = calc_annual_illiquidity_table_spd(df)  # by multiplying these values by 5, we get approximately the same result as the one in the paper




if __name__ == "__main__":
    main()


# draft ################################################################################
# dfv = pd.read_csv\
#     ('Volumes_BBW_TRACE_Enhanced_Dick_Nielsen.csv.gzip',
#      compression = "gzip")
# df = df.merge(dfv, how = "inner", left_on = ['TRD_EXCTN_DT','CUSIP_ID'],
#                                   right_on = ['TRD_EXCTN_DT','CUSIP_ID'])


# illiq = -cov(deltap, deltap_lag), groupby in month #
# Equation (2) in
# "The Illiquidity of Corporate Bonds" by Bao, Pan and Wang (2011)
# In The Journal of Finance
# Equation (2) in BBW's JFE Paper.

# tqdm.pandas()


# Illiq_month_all = df.groupby(['cusip', 'month_year'] )[['deltap','deltap_lag']]\
#     .progress_apply(lambda x: \
#                     x.cov().iloc[0,1]) * -1
# Illiq_month_all = Illiq_month_all.reset_index()
# Illiq_month_all.columns = ['cusip', 'month_year','illiq']

# Illiq_month_all = Illiq_month_all.sort_values(by='month_year')
# Illiq_month_all['year'] = Illiq_month_all['month_year'].dt.year
# Illiq_month_all = Illiq_month_all.dropna(subset=['illiq'])
# Illiq_daily_table3 = Illiq_month_all.groupby('year')['illiq'].mean().reset_index()
# # Illiq_daily_table3 = Illiq_month_all.groupby('year')['illiq'].mean().reset_index()
# Illiq_table_median3 = Illiq_month_all.groupby('year')['illiq'].median().reset_index()
# Illiq_month_all.describe()

# df = df.sort_values(by='month_year')
# Illiq_month = df.groupby(['cusip','month_year'] )[['deltap','deltap_lag']]\
#     .progress_apply(lambda x: \
#                     x.cov().iloc[0,1]) * -1
# Illiq_month = Illiq_month.reset_index()

# Illiq_month.columns = ['cusip','month_year','illiq']
# Illiq_month = Illiq_month.dropna(subset=['illiq'])
# Illiq_month['roll'] = np.where(Illiq_month['illiq'] >  0,
#                          (2 * np.sqrt(Illiq_month['illiq'])),
#                          0 )

# # Illiq_month = Illiq_month[Illiq_month['illiq'] < 1000]
# # lower_bound = Illiq_month['illiq'].quantile(0.05)
# # upper_bound = Illiq_month['illiq'].quantile(0.95)

# # Filter the DataFrame to keep only rows where 'illiq' is within the calculated bounds
# # Illiq_month_filtered = Illiq_month[(Illiq_month['illiq'] >= lower_bound) &
# #                                    (Illiq_month['illiq'] <= upper_bound)]


# Illiq_month['year'] = Illiq_month['month_year'].dt.year
# # Illiq_daily_table2 = Illiq_month.groupby(['cusip', 'year'])['illiq'].mean().reset_index()
# Illiq_daily_table2 = Illiq_month.groupby('year')['illiq'].mean().reset_index()

# np.mean(Illiq_month['illiq'])
# # Illiq['year'] = pd.to_datetime( Illiq['year'].astype(str) )
# # Illiq['date'] = Illiq['date'] + pd.offsets.MonthEnd(0)   
# Illiq_daily_table2_median = Illiq_month.groupby('year')['illiq'].median().reset_index()

# Illiq_daily_table_spd = Illiq_month.groupby('year')['roll'].mean().reset_index()


# # Calculate t-statistics for each cusip in each year
# Illiq_month['t_stat'] = Illiq_month.groupby(['month_year'])['illiq'].transform(
#     lambda x: (x.mean() / x.sem()) if x.sem() > 0 else np.nan
# )

# # Identify the entries with t-stat >= 1.96
# Illiq_month['significant'] = Illiq_month['t_stat'] >= 1.96

# # Calculate the percentage of significant t-stats for each year
# percent_significant = Illiq_month.groupby('year')['significant'].mean() * 100


# def get_robust_t_stat(group):
#     # Run OLS on a constant term only (mean of illiq) to get the intercept's t-stat
#     X = add_constant(group['illiq'])
#     ols_result = OLS(group['illiq'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    
#     # The t-stat for the intercept (mean of illiq) will be our robust t-stat
#     return abs(ols_result.tvalues[0])

# Illiq_month = Illiq_month.dropna(subset=['illiq', 't_stat'])

# # Calculate robust t-statistics for each year
# robust_t_stats = Illiq_month.groupby('year').apply(get_robust_t_stat)

# # Combine the results into a final DataFrame
# final_results = pd.DataFrame({
#     'Year': robust_t_stats.index,
#     'Mean_illiq': Illiq_month.groupby('year')['illiq'].mean(),
#     'Median_illiq': Illiq_month.groupby('year')['illiq'].median(),
#     'Per_t_greater_1_96': percent_significant,
#     'Robust_t_stat': robust_t_stats.values
# }).reset_index(drop=True)

# def calc_annual_illiquidity_table_spd(df):
#     """"""
#     df_unique = df.groupby(['cusip', 'month_year'])['t_spread'].first().reset_index()
#     df_unique['year'] = df_unique['month_year'].dt.year  
#     df_unique = df_unique.sort_values(by='month_year')
    
    
#     df_unique['spd_implied'] = np.power(df_unique['t_spread']/2, 2)
#     spd_implied_table = df_unique.groupby(['cusip', 'year'])['spd_implied'].mean()
#     spd_implied_table.describe()
    
#     # case 1
#     df_unique.groupby('year')['t_spread'].mean()
    
#     df_unique['logspread'] = np.log(df_unique['t_spread'])
#     df_unique['logspread_lag'] = df_unique.groupby('cusip')['logspread'].shift(1)
#     df_unique['delta_spread'] = df_unique['logspread'] - df_unique['logspread_lag']

#     # Restrict log returns to be in the interval [1,1]
#     df_unique['delta_spread'] = np.where(df_unique['delta_spread'] > 1, 1, df_unique['delta_spread'])
#     df_unique['delta_spread'] = np.where(df_unique['delta_spread'] <-1, -1, df_unique['delta_spread'])

#     # Convert logspread to % i.e. returns in % as opposed to decimals
#     df_unique['delta_spread'] = df_unique['delta_spread']

#     df_unique['logspread_lead'] = df_unique.groupby('cusip')['logspread'].shift(-1)
#     df_unique['delta_spread_lag'] = df_unique['logspread_lead'] - df_unique['logspread']
#     df_unique['delta_spread_lag'] = np.where(df_unique['delta_spread_lag'] > 1, 1,
#                                              df_unique['delta_spread_lag'])
#     df_unique['delta_spread_lag'] = np.where(df_unique['delta_spread_lag'] <-1, -1,
#                                              df_unique['delta_spread_lag'])
#     df_unique['delta_spread_lag'] = df_unique['delta_spread_lag']
    
#     df_unique = df_unique.dropna(subset=['delta_spread_lag', 'delta_spread', 't_spread'])
    
    
#     # case 2
#     df_unique = df_unique.dropna(subset=['t_spread'])
#     df_unique['spread_lag'] = df_unique.groupby('cusip')['t_spread'].shift(1)
#     df_unique['spread_lead'] = df_unique.groupby('cusip')['t_spread'].shift(-1)
#     df_unique['spread_lag'] = df_unique['spread_lag']
#     df_unique['spread_lead'] = df_unique['spread_lead']

#     df_unique['delta_s'] = (df_unique['t_spread'] - df_unique['spread_lag']) / df_unique['spread_lag']
#     df_unique['delta_s_lag'] = (df_unique['spread_lead'] - df_unique['t_spread']) / df_unique['t_spread']
#     df_unique = df_unique.dropna(subset=['delta_s', 'delta_s_lag', 't_spread'])
#     df_unique.describe()

#     Illiq_mean_table = df_unique.groupby(['cusip','year'] )[['delta_s','delta_s_lag']]\
#         .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
#     Illiq_mean_table = Illiq_mean_table.reset_index()
#     Illiq_mean_table.columns = ['cusip','year','illiq']
#     Illiq_mean_table.describe()
#     # Illiq_daily_annualbycusip = Illiq_month.groupby(['cusip', 'year'])['illiq'].mean().reset_index()
#     Illiq_mean_table = Illiq_mean_table.groupby('year')['illiq'].mean().reset_index()
#     overall_illiq_mean = np.mean(Illiq_month['illiq'])
#     Illiq_table_median = Illiq_month.groupby('year')['illiq'].median().reset_index()
#     overall_illiq_median = Illiq_month['illiq'].median()
    
#     df_unique = df_unique.dropna(subset=['delta_s', 'delta_s_lag', 't_spread'])

#     Illiq_mean_table = df_unique.groupby(['cusip','year'] )[['delta_s','delta_s_lag']]\
#         .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
#     Illiq_mean_table = Illiq_mean_table.reset_index()
#     Illiq_mean_table.columns = ['cusip','year','illiq']
#     Illiq_mean_table = Illiq_mean_table.dropna(subset=['illiq'])
#     Illiq_mean_table.describe()
#     # Illiq_daily_annualbycusip = Illiq_month.groupby(['cusip', 'year'])['illiq'].mean().reset_index()
#     Illiq_mean_table = Illiq_mean_table.groupby('year')['illiq'].mean().reset_index()
#     overall_illiq_mean = np.mean(Illiq_month['illiq'])
#     Illiq_table_median = Illiq_month.groupby('year')['illiq'].median().reset_index()
#     overall_illiq_median = Illiq_month['illiq'].median()
    
#     # case 3:
#     df_unique = df.groupby(['cusip', 'month_year'])['t_spread'].first().reset_index()
#     df_unique = df_unique.sort_values(by='month_year')
#     df_unique = df_unique.dropna(subset=['t_spread'])
#     df_unique['spread_lag'] = df_unique.groupby(['cusip'])['t_spread'].shift(1)
#     df_unique = df_unique.dropna(subset=['spread_lag'])
#     df_unique['year'] = df_unique['month_year'].dt.year
#     Illiq_mean_table = df_unique.groupby(['cusip','year'] )[['t_spread','spread_lag']]\
#         .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
#     Illiq_mean_table = Illiq_mean_table.reset_index()
#     Illiq_mean_table.columns = ['cusip','year','illiq']
    
#     # Drop NaN values that were produced by shifting
#     Illiq_mean_table = Illiq_mean_table.dropna(subset=['illiq'])
    
#     # Group by year and calculate the mean and median implied γ for each year
#     annual_illiq = Illiq_mean_table.groupby('year')['illiq'].agg(['mean', 'median']).reset_index()
    
#     # Calculate overall mean and median implied γ across the full sample period
#     overall_mean_gamma = df['gamma'].mean()
#     overall_median_gamma = df['gamma'].median()


    
#     df_unique.groupby('year')['t_spread'].mean()  
    
# cleaned_df.groupby('year')['t_spread'].median()
# df.groupby('year')['t_spread'].mean()
# df.groupby('year')['t_spread'].median()


# np.mean(cleaned_df['t_spread'])

################################################################################


# Price - Equal-Weight   #
# prc_EW = trace.groupby(['cusip_id','trd_exctn_dt'])[['rptd_pr']].mean().sort_index(level  =  'cusip_id').round(4) 
# prc_EW.columns = ['prc_ew']
        
# # Price - Volume-Weight # 
# trace['dollar_vol']    = ( trace['entrd_vol_qt'] * trace['rptd_pr']/100 ).round(0) # units x clean prc                               
# trace['value-weights'] = trace.groupby([ 'cusip_id','trd_exctn_dt'],
#                                                 group_keys=False)[['entrd_vol_qt']].apply( lambda x: x/np.nansum(x) )
# prc_VW = trace.groupby(['cusip_id','trd_exctn_dt'])[['rptd_pr','value-weights']].apply( lambda x: np.nansum( x['rptd_pr'] * x['value-weights']) ).to_frame().round(4)
# prc_VW.columns = ['prc_vw']
        
# PricesAll = prc_EW.merge(prc_VW, how = "inner", left_index = True, right_index = True)  
# PricesAll.columns                = ['prc_ew','prc_vw']              
# # Volume #
# VolumesAll                        = trace.groupby(['cusip_id','trd_exctn_dt'])[['entrd_vol_qt']].sum().sort_index(level  =  "cusip_id")                       
# VolumesAll['dollar_volume']       = trace.groupby(['cusip_id','trd_exctn_dt'])[['dollar_vol']].sum().sort_index(level  =  "cusip_id").round(0)
# VolumesAll.columns                = ['qvolume','dvolume'] 



# if __name__ == "__main__":
#     pass
