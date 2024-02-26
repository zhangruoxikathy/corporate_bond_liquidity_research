'''
Overview
-------------
This Python script aims to calculate illiquidity based on methodology in
The Illiquidity of Corporate Bonds, Bao, Pan, and Wang (2010).
 
Requirements
-------------

../data/pulled/Bondret.parquet resulting from load_wrds_bondret.py

Package versions 
-------------
pandas v1.4.4
numpy v1.21.5
wrds v3.1.2
datetime v3.9.13
csv v1.0
gzip v3.9.13
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

# df  = pd.read_csv('../data/manual/BondDailyPublic.csv.gzip',
#      compression='gzip')

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
    merged_df['trd_exctn_dt_lag'] = merged_df.\
        groupby('cusip')['trd_exctn_dt'].shift(1)
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


cleaned_df = clean_merged_data('2003-04-14', '2009-06-30')


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
    df['deltap_lag'] = df['deltap_lag'] * 100
    df['deltap_lag'] = np.where(df['deltap_lag'] > 1, 1, df['deltap_lag'])
    df['deltap_lag'] = np.where(df['deltap_lag'] <-1, -1, df['deltap_lag'])

    # df.isna().sum()
    # df_bondret.columns
    
    # Drop NAs in deltap, deltap_lag and bonds < 10 observations of the paired price changes
    df_final = df.dropna(subset=['deltap', 'deltap_lag', 'prclean'])  # 'offering_date', 'price_ldm', 'offering_price', 'amount_outstanding'])
    df_final['trade_counts'] = df_final.groupby(['cusip', 'year'])['deltap'].transform("count")
    df_final = df_final[df_final['trade_counts'] >= 10]    
    
    return df_final
    
    
df = calc_deltaprc(cleaned_df)


Illiq_month = df.groupby(['cusip','month_year'] )[['deltap','deltap_lag']]\
    .progress_apply(lambda x: \
                    x.cov().iloc[0,1]) * -1
Illiq_month = Illiq_month.reset_index()
Illiq_month.columns = ['cusip','month_year','illiq']

Illiq_month = Illiq_month.sort_values(by='month_year')
Illiq_month['year'] = Illiq_month['month_year'].dt.year
# Illiq_daily_table2 = Illiq_month.groupby(['cusip', 'year'])['illiq'].mean().reset_index()
Illiq_daily_table2 = Illiq_month.groupby('year')['illiq'].mean().reset_index()
np.mean(Illiq_month['illiq'])
# Illiq['year'] = pd.to_datetime( Illiq['year'].astype(str) )
# Illiq['date'] = Illiq['date'] + pd.offsets.MonthEnd(0)   
Illiq_daily_table2_median = Illiq_month.groupby('year')['illiq'].median().reset_index()



def calc_annual_illiquidity_table_daily(df):
    """Calculate illiquidity = -cov(deltap, deltap_lag), groupby in month. 
    """
    tqdm.pandas()
    
    Illiq_month = df.groupby(['cusip','month_year'] )[['deltap','deltap_lag']]\
        .progress_apply(lambda x: x.cov().iloc[0,1]) * -1
    Illiq_month = Illiq_month.reset_index()
    Illiq_month.columns = ['cusip','month_year','illiq']
    Illiq_month['year'] = Illiq_month['month_year'].dt.year
    # Illiq_daily_annualbycusip = Illiq_month.groupby(['cusip', 'year'])['illiq'].mean().reset_index()
    Illiq_mean_table = Illiq_month.groupby('year')['illiq'].mean().reset_index()
    overall_illiq_mean = np.mean(Illiq_month['illiq'])
    Illiq_table_median = Illiq_month.groupby('year')['illiq'].median().reset_index()
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
    table2_final = pd.DataFrame({
        'Year': robust_t_stats.index,
        'Mean_illiq': Illiq_month.groupby('year')['illiq'].mean(),
        'Median_illiq': Illiq_month.groupby('year')['illiq'].median(),
        'Per_t_greater_1_96': percent_significant,
        'Robust_t_stat': robust_t_stats.values
    }).reset_index(drop=True)
    
    overall_data = pd.DataFrame({
        'Year': ['Full'],
        'Mean_illiq': [overall_illiq_mean],
        'Median_illiq': [overall_illiq_median],
        'Per_t_greater_1_96': [overall_percent_significant],
        'Robust_t_stat': [overall_robust_t_stat]
    })

    table2_final = pd.concat([table2_final, overall_data], ignore_index=True)

    return table2_final


table2_final = calc_annual_illiquidity_table_daily(df)

latex_code = table2_final.to_latex(index=False, float_format="{:0.2f}".format, na_rep='NA')
print(latex_code)








# draft ################################################################################
# dfv = pd.read_csv\
#     ('Volumes_BBW_TRACE_Enhanced_Dick_Nielsen.csv.gzip',
#      compression = "gzip")
# df = df.merge(dfv, how = "inner", left_on = ['TRD_EXCTN_DT','CUSIP_ID'],
#                                   right_on = ['TRD_EXCTN_DT','CUSIP_ID'])

# df         = df.set_index(['CUSIP_ID', 'TRD_EXCTN_DT'])
unique_cusips_count = merged_df['cusip'].nunique()
merged_df = merged_df.sort_values(by='trd_exctn_dt')
# Log price
merged_df['logprc']     = np.log(merged_df['prclean'])
# Lag log price
merged_df['logprc_lag'] = merged_df.groupby( 'cusip' )['logprc'].shift(1)
# Difference in log prices #
merged_df['deltap']     = merged_df ['logprc'] - merged_df ['logprc_lag']

#* ************************************** */
#* Restrict log returns to be in the      */
#* interval [1,1]                         */
#* ************************************** */
# This trims some very extreme daily returns
# which could to do TRACE data errors etc.
# the trimming helps to give a more accurate
# value to the liquidity characteristics
# we compute 

merged_df['deltap'] = np.where(merged_df['deltap'] > 1, 1,
                        merged_df['deltap'])
merged_df['deltap'] = np.where(merged_df['deltap'] <-1, -1,
                        merged_df['deltap'])

# Convert deltap to % i.e. returns in % as opposed
# to decimals #
merged_df['deltap']     = merged_df['deltap']     * 100

# Lags days for day_counts
merged_df['trd_exctn_dt_lag'] = merged_df.\
    groupby('cusip')['trd_exctn_dt'].shift(1)


#####   check1 = Illiq_year[Illiq_year['illiq'] >= 20]

dfDC = merged_df.dropna(subset=['trd_exctn_dt_lag'])

# With U.S. Holidays #
# 2. Generate a list of holidays over this period
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
calendar = USFederalHolidayCalendar()
start_date = '01JUL2002'
end_date   = '31DEC2022'
holidays = calendar.holidays(start_date, end_date)
holidays
holiday_date_list = holidays.date.tolist()

dfDC['n']  = np.busday_count(dfDC['trd_exctn_dt_lag'].values.astype('M8[D]') , 
                                      dfDC['trd_exctn_dt'].values.astype('M8[D]'),
                                      holidays = holiday_date_list)

df = merged_df.merge(
              dfDC[['cusip',
                    'trd_exctn_dt',
                    'n']], left_on = ['cusip','trd_exctn_dt'],
                           right_on = ['cusip','trd_exctn_dt'] ,
                           how = "left")
del(dfDC)

# Lags days for day_counts
# df['deltap_lag'] = df.\
#     groupby( 'cusip')['deltap'].shift(1)

df['logprc_lead'] = df.groupby( 'cusip' )['logprc'].shift(-1)

df['deltap_lag']     = df ['logprc_lead'] - df ['logprc']
df['deltap_lag']     = df['deltap_lag']     * 100
df['deltap_lag'] = np.where(df['deltap_lag'] > 1, 1,
                        df['deltap_lag'])
df['deltap_lag'] = np.where(df['deltap_lag'] <-1, -1,
                        df['deltap_lag'])

# This follows Bao, Pan and Wang (BPW), 2011 in
# The Journal of Finance 
# BBW state in their paper they follow 
# the methodology of BPW

# only include "daily" return if the gap between 
# trades is less than 1-Week 
# Assumed to be 1-Week of Business days (they are unclear).
# See Footnote #10, of their JF Paper -- Page 918 of the Journal
# Page 8/36 of the .PDF file.

# df = df[df.n <= 7]
df.isna().sum()
# dffinal = df.dropna(subset=['deltap', 'deltap_lag'])
df_bondret.columns
dffinal = df.dropna(subset=['deltap', 'deltap_lag', 'prclean'])
# dffinal = df.dropna(subset=['deltap', 'deltap_lag', 'prclean', 'price_eom']) # 'price_ldm', 'offering_price', 'amount_outstanding'

dffinal['month_year']   = pd.to_datetime(dffinal['trd_exctn_dt']).dt.to_period('M') 
dffinal['trade_counts'] = dffinal.groupby(['cusip', 'year'] )['deltap'].transform("count")
dffinal = dffinal[dffinal['trade_counts'] >= 10]    
dffinal['abs_ret']      = np.abs(dffinal['deltap'])


# illiq = -cov(deltap, deltap_lag), groupby in month #
# Equation (2) in
# "The Illiquidity of Corporate Bonds" by Bao, Pan and Wang (2011)
# In The Journal of Finance
# Equation (2) in BBW's JFE Paper.

tqdm.pandas()


Illiq_month_all = df.groupby(['month_year'] )[['deltap','deltap_lag']]\
    .progress_apply(lambda x: \
                    x.cov().iloc[0,1]) * -1
Illiq_month_all = Illiq_month_all.reset_index()
Illiq_month_all.columns = ['month_year','illiq']

Illiq_month_all = Illiq_month_all.sort_values(by='month_year')
Illiq_month_all['year'] = Illiq_month_all['month_year'].dt.year
Illiq_daily_table3 = Illiq_month_all.groupby('year')['illiq'].mean().reset_index()
# Illiq_daily_table3 = Illiq_month_all.groupby('year')['illiq'].mean().reset_index()



Illiq_month = df.groupby(['cusip','month_year'] )[['deltap','deltap_lag']]\
    .progress_apply(lambda x: \
                    x.cov().iloc[0,1]) * -1
Illiq_month = Illiq_month.reset_index()
Illiq_month.columns = ['cusip','month_year','illiq']

Illiq_month = Illiq_month.sort_values(by='month_year')
Illiq_month['year'] = Illiq_month['month_year'].dt.year
# Illiq_daily_table2 = Illiq_month.groupby(['cusip', 'year'])['illiq'].mean().reset_index()
Illiq_daily_table2 = Illiq_month.groupby('year')['illiq'].mean().reset_index()
np.mean(Illiq_month['illiq'])
# Illiq['year'] = pd.to_datetime( Illiq['year'].astype(str) )
# Illiq['date'] = Illiq['date'] + pd.offsets.MonthEnd(0)   
Illiq_daily_table2_median = Illiq_month.groupby('year')['illiq'].median().reset_index()




# Calculate t-statistics for each cusip in each year
Illiq_month['t_stat'] = Illiq_month.groupby(['cusip', 'year'])['illiq'].transform(
    lambda x: (x.mean() / x.sem()) if x.sem() > 0 else np.nan
)

# Identify the entries with t-stat >= 1.96
Illiq_month['significant'] = Illiq_month['t_stat'] >= 1.96

# Calculate the percentage of significant t-stats for each year
percent_significant = Illiq_month.groupby('year')['significant'].mean() * 100




def get_robust_t_stat(group):
    # Run OLS on a constant term only (mean of illiq) to get the intercept's t-stat
    X = add_constant(group['illiq'])
    ols_result = OLS(group['illiq'], X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    
    # The t-stat for the intercept (mean of illiq) will be our robust t-stat
    return abs(ols_result.tvalues[0])

Illiq_month = Illiq_month.dropna(subset=['illiq', 't_stat'])

# Calculate robust t-statistics for each year
robust_t_stats = Illiq_month.groupby('year').apply(get_robust_t_stat)

# Combine the results into a final DataFrame
final_results = pd.DataFrame({
    'Year': robust_t_stats.index,
    'Mean_illiq': Illiq_month.groupby('year')['illiq'].mean(),
    'Median_illiq': Illiq_month.groupby('year')['illiq'].median(),
    'Per_t_greater_1_96': percent_significant,
    'Robust_t_stat': robust_t_stats.values
}).reset_index(drop=True)



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



if __name__ == "__main__":
    pass
