
'''
Overview
-------------
This Python script aims to .
 
Requirements
-------------

../data/pulled/Bondret.parquet resulting from load_wrds.py

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
import numpy as np
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import misc_tools
import load_wrds_bondret
import load_opensource
import data_processing as data

# df  = pd.read_csv('../data/manual/BondDailyPublic.csv.gzip',
#      compression='gzip')
df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)
df_bondret = load_wrds_bondret.load_bondret(data_dir=DATA_DIR)
merged_df = data.all_trace_data_merge(df_daily, df_bondret, start_date = '2003-04-14', end_date = '2009-06-30')
merged_df = data.sample_selection(merged_df, start_date = '2003-04-14', end_date = '2009-06-30')


# Log price
merged_df['logprc']     = np.log(merged_df['prclean'])
# Lag log price
merged_df['logprc_lag'] = merged_df.groupby( 'cusip' )['logprc'].shift(1)
# Difference in log prices #
merged_df['deltap']     = merged_df ['logprc'] - merged_df ['logprc_lag']

# This trims some very extreme daily returns
# which could to do TRACE data errors etc.
# the trimming helps to give a more accurate
# value to the liquidity characteristics
# we compute 

# merged_df['deltap'] = np.where(merged_df['deltap'] > 1, 1,
#                         merged_df['deltap'])
# merged_df['deltap'] = np.where(merged_df['deltap'] <-1, -1,
#                         merged_df['deltap'])

# Convert deltap to % i.e. returns in % as opposed
# to decimals #
merged_df['deltap']     = merged_df['deltap']     * 100

# Lags days for day_counts
merged_df['trd_exctn_dt_lag'] = merged_df.\
    groupby('cusip')['trd_exctn_dt'].shift(1)


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
df['deltap_lag'] = df.\
    groupby( 'cusip')['deltap'].shift(-1)

# This follows Bao, Pan and Wang (BPW), 2011 in
# The Journal of Finance 
# BBW state in their paper they follow 
# the methodology of BPW

# only include "daily" return if the gap between 
# trades is less than 1-Week 
# Assumed to be 1-Week of Business days (they are unclear).
# See Footnote #10, of their JF Paper -- Page 918 of the Journal
# Page 8/36 of the .PDF file.

df = df[df.n <= 7]
dffinal = df.dropna(subset=['deltap', 'deltap_lag'])

dffinal['month_year']   = pd.to_datetime(dffinal['trd_exctn_dt']).dt.to_period('M') 
dffinal['trade_counts'] = dffinal.groupby(['cusip', 'year'] )['deltap'].transform("count")
dffinal = dffinal[dffinal['trade_counts'] >= 10]    
dffinal['abs_ret']      = np.abs(dffinal['deltap'])


# illiq = -cov(deltap, deltap_lag), groupby in month #
# Equation (2) in
# "The Illiquidity of Corporate Bonds" by Bao, Pan and Wang (2011)
# In The Journal of Finance
# Equation (2) in BBW's JFE Paper.

from tqdm import tqdm

tqdm.pandas()

# # Define a function to calculate the covariance and multiply by -1
# def calculate_cov(x):
#     return x.cov().iloc[0, 1] * -1

# # Apply the function with progress_apply
# Illiq = dffinal.groupby(['cusip', 'year'])[['deltap', 'deltap_lag']].progress_apply(calculate_cov)

Illiq = dffinal.groupby(['cusip','year'] )[['deltap','deltap_lag']]\
    .progress_apply(lambda x: \
                    x.cov().iloc[0,1]) * -1
    
Illiq = Illiq.reset_index()
Illiq.columns = ['cusip','year','illiq']
# Illiq['year'] = pd.to_datetime( Illiq['year'].astype(str) )
# Illiq['date'] = Illiq['date'] + pd.offsets.MonthEnd(0)   
Illiq['roll'] = np.where(Illiq['illiq'] >  0,
                         (2 * np.sqrt(Illiq['illiq'])),
                         0 )

Illiq.groupby('year')['illiq'].mean()






# daily data clean attempt
# df_daily['trd_exctn_dt'] = pd.to_datetime(df_daily['trd_exctn_dt'])
# cutoff_date = pd.Timestamp('2009-06-30')
# df = df_daily[df_daily['trd_exctn_dt'] <= cutoff_date].dropna()
# periods = df.groupby('cusip_id')['trd_exctn_dt'].agg(['max', 'min'])
# periods['max_period'] = (periods['max'] - periods['min']).dt.days
# periods['threshold'] = periods['max_period'] * 0.75 * 252 / 365.25
# counts = df['cusip_id'].value_counts()
# to_keep_days = counts[counts > 10]
# ids_to_keep = counts[counts > periods.loc[counts.index, 'threshold']]
# df_filtered = df[df['cusip_id'].isin(ids_to_keep.index) & df['cusip_id'].isin(to_keep_days.index)]


# df_2004 = df_filtered[df_filtered['trd_exctn_dt'].dt.year == 2004]
# df_2004['cusip_id'].nunique()





# monthly data clean attempt
# Calculate via monthly data
df_bondret = load_wrds_bondret.load_bondret(data_dir=DATA_DIR)
df_bondret['logprc']     = np.log(df_bondret['price_eom'])
df_bondret = df_bondret.sort_values(['cusip', 'date'])
df_bondret['deltap'] = df_bondret.groupby('cusip')['logprc'].diff().round(5)
df_bondret['deltap_lag'] = df_bondret.groupby('cusip')['logprc'].shift(
    -1) - df_bondret['logprc'].round(5)
df_bondret_cleaned = df_bondret.dropna(subset=['deltap', 'deltap_lag'])

def negative_cov(x):
    cov_matrix = x[['deltap', 'deltap_lag']].cov()
    neg_cov = -cov_matrix.loc['deltap', 'deltap_lag']
    return neg_cov if neg_cov < 0 else 0


average_neg_cov = df_bondret_cleaned.groupby(['cusip', 'year']).apply(
    negative_cov).groupby(level=1).mean()



if __name__ == "__main__":
    pass
