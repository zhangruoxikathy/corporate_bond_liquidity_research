
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

df_bondret = load_wrds_k.load_bondret(data_dir=DATA_DIR)
df_bondret['logprc']     = np.log(df_bondret['offering_price'])
df_bondret = df_bondret.sort_values(['cusip', 'date'])
df_bondret['deltap'] = df_bondret.groupby('cusip')['offering_price'].diff().round(5)
df_bondret['deltap_lag'] = df_bondret.groupby('cusip')['offering_price'].shift(
    -1) - df_bondret['offering_price'].round(5)

df_bondret_cusip = df_bondret.groupby('cusip')['offering_price'].mean()

df_bondret['deltap_lag'] = df_bondret.

if __name__ == "__main__":
    pass
