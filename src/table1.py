import pandas as pd
import config
import load_wrds_k

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

df = load_wrds_k.load_bondret(data_dir = DATA_DIR)

def sample_data(df):

    # select data from 2003-04-14 to 2009-6-30 
    df['date'] = pd.to_datetime(df['date'])
    start_date = '2003-04-14'
    end_date = '2009-06-30'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # drop all bonds that only exist after the date of phase 3: Feb 7 2005

    cutoff_date = pd.Timestamp('2005-02-07')

    df = df.groupby('cusip').filter(lambda x: x['date'].min() <= cutoff_date)

    # make sure it exist for at least one full year 
    df = df.groupby('cusip').filter(lambda x: x['date'].max() - x['date'].min() >= pd.Timedelta(days=365))

    # keep only investment grade
    df = df[df['rating_num'] <= 10]

    return df 


df_filtered = sample_data(df)

print("Number of bonds in the sample: " + str(len(df_filtered['cusip'].unique())))

# gives an output of 6627   vs  1035 in the paper!!! 