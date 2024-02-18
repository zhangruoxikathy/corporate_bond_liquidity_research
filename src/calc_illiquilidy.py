
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
import load_wrds_k
import load_opensource

# df  = pd.read_csv('../data/manual/BondDailyPublic.csv.gzip',
#      compression='gzip')
df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)
df_daily['trd_exctn_dt'] = pd.to_datetime(df_daily['trd_exctn_dt'])
df_2004 = df_daily[df_daily['trd_exctn_dt'].dt.year == 2004]
df_2004['cusip_id'].nunique()





# Calculate via monthly data
df_bondret = load_wrds_k.load_bondret(data_dir=DATA_DIR)
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
