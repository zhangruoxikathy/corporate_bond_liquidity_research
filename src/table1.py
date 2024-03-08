'''
Overview
-------------
This Python script aims to replicate table 1 summary statistics in the paper with
periods in the paper and update it to the present.

Requirements
-------------

../data/pulled/Bondret resulting from load_wrds_bondret.py
../data/pulled/BondDailyPublic resulting from load_opensource.py
../data/pulled/IntradayTRACE resulting from load_intraday.py
'''

import pandas as pd
import numpy as np
import config
import load_wrds_bondret
import load_opensource
import data_processing
import load_intraday
import datetime

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR
START_DATE = config.START_DATE
END_DATE = config.END_DATE



def cal_avrage(dataframe, column):
    """
    Calculate the average of a specified column in a dataframe grouped by year.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    column (str): The column name for which the average is calculated.

    Returns:
    pandas.Series: A series containing the average values for each year.
    """
    average = dataframe.groupby('year')[column].mean().reset_index()
    average.rename(columns={column: column+'_avg'}, inplace=True)
    average.set_index('year', inplace=True)

    return average

def cal_median(dataframe, column):
    """
    Calculate the median value of a specified column in a dataframe, grouped by year.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    column (str): The name of the column to calculate the median for.

    Returns:
    pandas.Series: A series containing the median values for each year.
    """
    median = dataframe.groupby('year')[column].median().reset_index()
    median.rename(columns={column: column+'_median'}, inplace=True)
    median.set_index('year', inplace=True)

    return median

def cal_std(dataframe, column):
    """
    Calculate the standard deviation of a column in a dataframe grouped by year.

    Args:
        dataframe (pandas.DataFrame): The input dataframe.
        column (str): The name of the column to calculate the standard deviation for.

    Returns:
        pandas.Series: The standard deviation of the specified column grouped by year.
    """
    std = dataframe.groupby('year')[column].std().reset_index()
    std.rename(columns={column: column+'_std'}, inplace=True)
    std.set_index('year', inplace=True)

    return std

def cal_count(dataframe, column='cusip'):
    """
    Calculate the count of unique values in a specified column of a dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    column (str): The column name to calculate the count of unique values. Default is 'cusip'.

    Returns:
    pandas.Series: A series containing the count of unique values for each year.
    """
    count = dataframe.groupby('year')[column].nunique().reset_index()
    count.rename(columns={column: column+'_count'}, inplace=True)
    count.set_index('year', inplace=True)
    return count

def calculation(df_sample, df_all, df_intraday):

    # Give df_all and df_sample a month column
    df_all['month'] = df_all['date'].dt.month
    df_sample['month'] = df_sample['date'].dt.month

    # merge the df_intraday_grouped with df_sample and df_all by year, month and cusip, only keep the #trade column
    df_intraday.rename(columns={'cusip_id': 'cusip'}, inplace=True)
    df_intraday['trd_exctn_dt'] = pd.to_datetime(df_intraday['trd_exctn_dt'])
    df_intraday['year'] = df_intraday['trd_exctn_dt'].dt.year
    df_intraday['month'] = df_intraday['trd_exctn_dt'].dt.month
    df_intraday_grouped = df_intraday.groupby(['year', 'month', 'cusip'])['trd_exctn_dt'].count().reset_index(name='#trade')

    del df_intraday

    df_sample = pd.merge(df_sample, df_intraday_grouped, how='left', on=['year', 'month', 'cusip'])

    df_all = pd.merge(df_all, df_intraday_grouped, how='left', on=['year', 'month', 'cusip'])

    # Calculate the number of unique cusips in df_sample and df_all
    df_sample_cusip = cal_count(df_sample)
    df_all_cusip = cal_count(df_all)


    # Calculate the Issuance of df_sample and df_all
    df_sample['issuance'] = df_sample['offering_amt'] * df_sample['principal_amt'] * \
                            df_sample['offering_price'] / 100 / 1000000

    df_sample_issuance = pd.concat([cal_avrage(df_sample, 'issuance'), \
                        cal_median(df_sample, 'issuance'), cal_std(df_sample, 'issuance')], axis=1)

    df_all['issuance'] = df_all['offering_amt'] * df_all['principal_amt'] * \
                        df_all['offering_price'] / 100 / 1000000

    df_all_issuance = pd.concat([cal_avrage(df_all, 'issuance'), \
                        cal_median(df_all, 'issuance'), cal_std(df_all, 'issuance')], axis=1)

    # Calculate the Moondy Rating of df_sample and df_all
    df_sample_moody = pd.concat([cal_avrage(df_sample, 'n_mr'), \
                        cal_median(df_sample, 'n_mr'), cal_std(df_sample, 'n_mr')], axis=1)

    df_all_moody = pd.concat([cal_avrage(df_all, 'n_mr'), \
                        cal_median(df_all, 'n_mr'), cal_std(df_all, 'n_mr')], axis=1)

    # Calculate the Maturity of df_sample and df_all
    df_sample_maturity = pd.concat([cal_avrage(df_sample, 'tmt'), \
                        cal_median(df_sample, 'tmt'), cal_std(df_sample, 'tmt')], axis=1)

    df_all_maturity = pd.concat([cal_avrage(df_all, 'tmt'), \
                        cal_median(df_all, 'tmt'), cal_std(df_all, 'tmt')], axis=1)

    # Calculate the coupon of df_sample and df_all
    df_sample_coupon = pd.concat([cal_avrage(df_sample, 'coupon'), \
                        cal_median(df_sample, 'coupon'), cal_std(df_sample, 'coupon')], axis=1)

    df_all_coupon = pd.concat([cal_avrage(df_all, 'coupon'), \
                        cal_median(df_all, 'coupon'), cal_std(df_all, 'coupon')], axis=1)

    # Calculate the age where the gap between the issuance date and the trade date in years
    df_sample[['date', 'offering_date']] = df_sample[['date', 'offering_date']].apply(pd.to_datetime)
    df_all[['date', 'offering_date']] = df_all[['date', 'offering_date']].apply(pd.to_datetime)

    df_sample['age'] = (df_sample['date'] - df_sample['offering_date']).dt.days / 365
    df_all['age'] = (df_all['date'] - df_all['offering_date']).dt.days / 365

    df_sample_age = pd.concat([cal_avrage(df_sample, 'age'), \
                        cal_median(df_sample, 'age'), cal_std(df_sample, 'age')], axis=1)

    df_all_age = pd.concat([cal_avrage(df_all, 'age'), \
                        cal_median(df_all, 'age'), cal_std(df_all, 'age')], axis=1)

    # Calculate the turnover in df_sample and df_all
    df_sample['turnover'] = df_sample['t_volume'] / df_sample['issuance'] / 10000
    df_all['turnover'] = df_all['t_volume'] / df_all['issuance'] / 10000

    df_sample_turnover = pd.concat([cal_avrage(df_sample, 'turnover'), \
                        cal_median(df_sample, 'turnover'), cal_std(df_sample, 'turnover')], axis=1)

    df_all_turnover = pd.concat([cal_avrage(df_all, 'turnover'), \
                        cal_median(df_all, 'turnover'), cal_std(df_all, 'turnover')], axis=1)


    # Calculate the return of df_sample and df_all

    # We need to drop teh duplicate entires in df_sample and df_all
    df_sample_month = df_sample.drop_duplicates(subset=['cusip', 'year', 'month']).reset_index()
    df_all_month = df_all.drop_duplicates(subset=['cusip', 'year', 'month']).reset_index()

    df_sample_month['return'] = np.log(df_sample_month['price_eom'] / \
                                df_sample_month.groupby(['cusip'])['price_eom'].shift(1)) * 100

    df_all_month['return'] = np.log(df_all_month['price_eom'] / \
                                df_all_month.groupby(['cusip'])['price_eom'].shift(1)) * 100


    # group by year and cusip and calculate the weighted average return
    df_sample_month_grouped = df_sample_month.groupby(['year', 'cusip'])['return'].mean().reset_index(name='Avg_return')
    df_all_month_grouped = df_all_month.groupby(['year', 'cusip'])['return'].mean().reset_index(name='Avg_return')


    df_sample_return = pd.concat([cal_avrage(df_sample_month_grouped, 'Avg_return'), \
                        cal_median(df_sample_month_grouped, 'Avg_return'), cal_std(df_sample_month_grouped, 'Avg_return')], axis=1)

    df_all_return = pd.concat([cal_avrage(df_all_month_grouped, 'Avg_return'), \
                        cal_median(df_all_month_grouped, 'Avg_return'), cal_std(df_all_month_grouped, 'Avg_return')], axis=1)

    # Calculate the volatility of df_sample and df_all

    df_sample_vol_grouped = df_sample_month.groupby(['year', 'cusip'])['return'].std().reset_index(name='volatility')
    df_all_vol_grouped = df_all_month.groupby(['year', 'cusip'])['return'].std().reset_index(name='volatility')

    df_sample_vol = pd.concat([cal_avrage(df_sample_vol_grouped, 'volatility'), \
                        cal_median(df_sample_vol_grouped, 'volatility'), cal_std(df_sample_vol_grouped, 'volatility')], axis=1)

    df_all_vol = pd.concat([cal_avrage(df_all_vol_grouped, 'volatility'), \
                        cal_median(df_all_vol_grouped, 'volatility'), cal_std(df_all_vol_grouped, 'volatility')], axis=1)

    # Calculate the Price in df_sample and df_all
    df_sample_month_price = df_sample.groupby(['year', 'cusip', 'date'])['prclean'].mean().reset_index()
    df_all_month_price = df_all.groupby(['year', 'cusip', 'date'])['prclean'].mean().reset_index()

    df_sample_price = pd.concat([cal_avrage(df_sample_month_price, 'prclean'), \
                        cal_median(df_sample_month_price, 'prclean'), cal_std(df_sample_month_price, 'prclean')], axis=1)

    df_all_price = pd.concat([cal_avrage(df_all_month_price, 'prclean'), \
                        cal_median(df_all_month_price, 'prclean'), cal_std(df_all_month_price, 'prclean')], axis=1)

    # Calculate the number of trades in df_sample and df_all
    df_sample_trade = pd.concat([cal_avrage(df_sample_month, '#trade'), \
                        cal_median(df_sample_month, '#trade'), cal_std(df_sample_month, '#trade')], axis=1)

    df_all_trade = pd.concat([cal_avrage(df_all_month, '#trade'), \
                        cal_median(df_all_month, '#trade'), cal_std(df_all_month, '#trade')], axis=1)


    # Calculate the Trade_size in df_sample and df_all
    df_sample_month['trade_size'] = df_sample_month['t_dvolume'] / df_sample_month['#trade'] / 1000
    df_all_month['trade_size'] = df_all_month['t_dvolume'] / df_all_month['#trade'] / 1000

    df_sample_size = pd.concat([cal_avrage(df_sample_month, 'trade_size'), \
                        cal_median(df_sample_month, 'trade_size'), cal_std(df_sample_month, 'trade_size')], axis=1)

    df_all_size = pd.concat([cal_avrage(df_all_month, 'trade_size'), \
                        cal_median(df_all_month, 'trade_size'), cal_std(df_all_month, 'trade_size')], axis=1)



    # concat all results of df_sample and df_all
    df_sample_result = pd.concat([df_sample_cusip, df_sample_issuance, df_sample_moody, df_sample_maturity, df_sample_coupon, \
                        df_sample_age, df_sample_turnover, df_sample_size, df_sample_trade, df_sample_return, df_sample_vol, df_sample_price], axis=1)

    df_all_result = pd.concat([df_all_cusip, df_all_issuance, df_all_moody, df_all_maturity, df_all_coupon, \
                        df_all_age, df_all_turnover, df_all_size, df_all_trade, df_all_return, df_all_vol, df_all_price], axis=1)
    # transform the df_sample_result, make its index as column
    df_sample_result = df_sample_result.T
    df_all_result = df_all_result.T

    return df_sample_result, df_all_result

if __name__ == "__main__":


    # #loading raw data 
    df_bondret = load_wrds_bondret.load_bondret(data_dir = DATA_DIR)
    df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)
    df_intraday = load_intraday.load_intraday_TRACE(data_dir=DATA_DIR, start_date = START_DATE, end_date = END_DATE)

    # pre-processing the data
    df_all = data_processing.all_trace_data_merge(df_daily, df_bondret)   #this is the dataset for panel B in table 1 
    del df_daily, df_bondret
    df_sample = data_processing.sample_selection(df_all) # this is the dataset for panel A in table 1

    df_sample_result, df_all_result = calculation(df_sample, df_all, df_intraday)
    df_sample_result.to_csv(OUTPUT_DIR / "table1_panelA.csv")
    df_all_result.to_csv(OUTPUT_DIR / "table1_panelB.csv")
    del df_sample_result, df_all_result, df_intraday, df_sample, df_all


    # #loading raw data 
    # today = datetime.today().strftime('%Y-%m-%d')
    df_bondret = load_wrds_bondret.load_bondret(data_dir = DATA_DIR)
    df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)
    df_intraday = load_intraday.load_intraday_TRACE(data_dir=DATA_DIR, start_date=START_DATE, end_date='2023-12-31')

    df_all_uptodate = data_processing.all_trace_data_merge(df_daily, df_bondret, start_date=START_DATE, end_date='2023-12-31')   #this is the dataset for panel B in table 1 
    del df_daily, df_bondret
    df_sample_uptodate = data_processing.sample_selection(df_all_uptodate, start_date=START_DATE, end_date='2023-12-31') # this is the dataset for panel A in table 1

    
    df_sample_result_uptodate, df_all_result_uptodate = calculation(df_sample_uptodate, df_all_uptodate, df_intraday)
    df_sample_result_uptodate.to_csv(OUTPUT_DIR / "table1_panelA_uptodate.csv")
    df_all_result_uptodate.to_csv(OUTPUT_DIR / "table1_panelB_uptodate.csv")
    del df_sample_result_uptodate, df_all_result_uptodate, df_intraday, df_sample_uptodate, df_all_uptodate



