'''
this file designs two function to propcess the data for producing table 1 and and 2

1) all_trace_data_merge function: it merges the daily TRARCE opensource data downloaded from https://openbondassetpricing.com/  
    with the montly Bondret data from WRDS based the same on cusip and time 

2) sample_selection function: this function selects samples to be included in paper following the exact steps as outlined in the paper 

'''

import pandas as pd
# import dask.dataframe as dd


def all_trace_data_merge(df_daily, df_bondret, start_date = '2003-04-14', end_date = '2009-06-30'):
    
    """
    this function merges the daily TRARCE opensource data downloaded from https://openbondassetpricing.com/  
    with the montly Bondret data from WRDS based the same on cusip and time 
    """

    # keep only the portion within select time
    # df_daily = df_daily.copy()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_daily = df_daily[['cusip_id', 'trd_exctn_dt', 'prclean']]

    # df_daily['trd_exctn_dt'] = pd.to_datetime(df_daily['trd_exctn_dt'])

    df_daily = df_daily[(df_daily['trd_exctn_dt'] >= start_date) & (df_daily['trd_exctn_dt'] <= end_date)]

    # df_daily['trd_exctn_dt'] = pd.to_datetime(df_daily['trd_exctn_dt'])
    
    #create a new column "month_time" based on which we do the merge 

    # ddf_daily = dd.from_pandas(df_daily, npartitions=8)
    # ddf_daily['month_time'] = ddf_daily['trd_exctn_dt'].dt.strftime('%Y-%m')
    # df_daily = ddf_daily.compute()
    df_daily['month_time'] = df_daily['trd_exctn_dt'].dt.strftime('%Y-%m')
    
    df_daily.rename(columns={'cusip_id': 'cusip'}, inplace=True)
    
    # df_bondret['date'] = pd.to_datetime(df_bondret['date'])
    df_bondret['month_time'] = df_bondret['date'].dt.strftime('%Y-%m')
    
    #with this merge methodology, we need to assume that all time-dependent variables from bondret remains unchanged within each month


    merged_df = pd.merge(df_daily, df_bondret, how='left', on=['cusip', 'month_time'])
    
    #adjust year based on trace data, replace the original 'year' column since it is from bondret and may contrain NA
    merged_df['year'] = merged_df['trd_exctn_dt'].dt.year
    
    return merged_df



def sample_selection(df, start_date = '2003-04-14', end_date = '2009-06-30'):

    """
    this function selects samples to be included in paper following the below steps as outlined in the paper 
    """

    # select Phase I and II bonds from 2003-04-14 to 2009-6-30 
    #start_date = '2003-04-14'
    #end_date = '2009-06-30'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['trd_exctn_dt'] >= start_date) & (df['trd_exctn_dt'] <= end_date)]
    
    # drop all bonds that only exist after the date of phase 3: Feb 7 2005
    cutoff_date = pd.Timestamp('2005-02-07')
    df = df.groupby('cusip').filter(lambda x: x['trd_exctn_dt'].min() <= cutoff_date)

    # make sure a cusip trade on at least 75% of its relevant business days
    periods = df.groupby('cusip')['trd_exctn_dt'].agg(['max', 'min'])
    periods['max_period'] = (periods['max'] - periods['min']).dt.days # maximum calendar days for exsitence
    periods['threshold'] = periods['max_period'] * 0.75 * 252 / 365  # 252/365 reprsents approximate business days proportion
    counts = df['cusip'].value_counts()
    ids_to_keep = counts[counts > periods.loc[counts.index, 'threshold']]
    #make sure the bonds are traded in more than 11 days to have 10 observations of (pt, p(t-1))
    to_keep_days = counts[counts > 10]
    df = df[df['cusip'].isin(ids_to_keep.index) & df['cusip'].isin(to_keep_days.index)]

    # make sure it exist for at least one full year 
    df = df.groupby('cusip').filter(lambda x: x['trd_exctn_dt'].max() - x['trd_exctn_dt'].min() >= pd.Timedelta(days=365))

    #drop all non investment-grade bonds using moody's rating; we do need to keep those na since they are incomplete infor from Bondret
    df = df[(df['n_mr'] <= 10) | (df['n_mr'].isna())]
    
    #adjust year based on trace, replace the original 'year' column since it is from bondret and may contrain NA
    df['year'] = df['trd_exctn_dt'].dt.year
    
    
    return df


