import pandas as pd
import config
import load_wrds_bondret
import load_opensource
import data_processing

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

#loading raw data 
df_bondret = load_wrds_bondret.load_bondret(data_dir = DATA_DIR)
df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)

#processing data 
df_all = data_processing.all_trace_data_merge(df_daily, df_bondret)   #this is the dataset for panel B in table 1 
df_sample = data_processing.sample_selection(df_all) # this is the dataset for panel A in table 1


def bond_number(df, year):
    
    df_select = df[df['year'] == year]
    return df_select['cusip'].nunique()


def sum_stat(df, year, variable):
    
    df_select = df[df['year'] == year]
    
    mean = df_select[variable].mean()
    median  = df_select[variable].median()
    sd = df_select[variable].std()
    
    return mean, median, sd


#calculate summary statistics in sample for panel A
bond_num_sample = {}
summary_rating_sample = {} #rating
summary_coupon_sample = {} #coupon
summary_maturity_sample = {} #maturity

for year in range(2003, 2010): 
    bond_num_sample[year] = bond_number(df_sample, year)
    
    
for year in range(2003, 2010): 
    summary_rating_sample[year] = sum_stat(df_sample, year, 'n_mr')
    summary_coupon_sample[year] = sum_stat(df_sample, year, 'coupon_x')
    summary_maturity_sample[year] = sum_stat(df_sample, year,'tmt')




#calculate summary statistics for all data for panel B 

bond_num_all = {}
summary_rating_all = {} #rating
summary_coupon_all = {} #coupon
summary_maturity_all = {} #maturity

for year in range(2003, 2010): 
    bond_num_all[year] = bond_number(df_all, year)
for year in range(2003, 2010): 
    summary_rating_all[year] = sum_stat(df_all, year,'n_mr')
    summary_coupon_all[year] = sum_stat(df_all, year, 'coupon_x')
    summary_maturity_all[year] = sum_stat(df_all, year,'tmt')


print(bond_num_sample)
print(summary_rating_sample)
print(summary_coupon_sample)
print(summary_maturity_sample)


print(bond_num_all)
print(summary_rating_all)
print(summary_coupon_all)
print(summary_maturity_all)
