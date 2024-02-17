import pandas as pd
import config
import load_wrds_k

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

df = load_wrds_k.load_bondret(data_dir = DATA_DIR)

def sample_data(df):

    # following the data clean setups in the paper 

    # select Phase I and II bonds from 2003-04-14 to 2009-6-30 
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
    df_sample = df[df['rating_num'] <= 10]

    return df_sample


df_sample = sample_data(df)

df_sample_2003 = df_sample[df_sample['year'] == 2003]
df_all_2003 = df[df['year'] == 2003 ]


print("Number of bonds in the sample in 2003: " + str(len(df_sample_2003['cusip'].unique())))
print("Number of all bonds in 2003: " + str(len(df_all_2003['cusip'].unique())))

# gives an output of 5560 and 9875, respectively;  in this paper they are 744 and 4161, respectively