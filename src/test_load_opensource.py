'''
Overview
-------------
This Python script contains the unit tests for data extracted from https://openbondassetpricing.com/
via load_opensource.py, which are daily bond data and MMN corrected monthly bond data.


Requirements
-------------

../data/pulled/BondDailyPublic.parquet resulting from load_opensource.py
../data/pulled/WRDS_MMN_Corrected_Data.csv.gzip resulting from load_opensource.py
../src/load_opensource.py

'''

import pandas as pd
import numpy as np
import pytest
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import load_opensource

df_daily = load_opensource.load_daily_bond(data_dir=DATA_DIR)
df_mmn = load_opensource.load_mmn_corrected_bond(data_dir=DATA_DIR)


def test_load_daily_bond_functionality():
    """Test the functionality of daily bond data loaded from Open Source Bond Asset Pricing."""

    # Test if the function returns a pandas DataFrame
    assert isinstance(df_daily, pd.DataFrame)

    # Test if the DataFrame has the expected columns
    expected_columns = ['cusip_id', 'trd_exctn_dt', 'prclean', 'qvolume']
    assert all(col in df_daily.columns for col in expected_columns)

    # Test if the function raises an error when given an invalid data directory
    with pytest.raises(FileNotFoundError):
        load_opensource.load_daily_bond(data_dir="invalid_directory")


def test_load_daily_bond_data_validity():
    """Test the validity of daily bond data loaded from Open Source Bond Asset Pricing."""

    # Test if the default date range has the expected start date and end date
    assert df_daily['trd_exctn_dt'].min() == pd.Timestamp('2002-07-01')
    assert df_daily['trd_exctn_dt'].max() >= pd.Timestamp('2022-12-01')

    # Test one year of summary statisticas of daily bond data
    df_daily['trd_exctn_dt'] = pd.to_datetime(df_daily['trd_exctn_dt'])
    df_daily['year'] = df_daily['trd_exctn_dt'].dt.year
    df_daily_sample = df_daily[df_daily['year'] == 2005]
    
    output_shape = df_daily_sample.shape
    expected_shape = (826209, 17)

    output = df_daily_sample[['cusip_id', 'trd_exctn_dt', 'prclean', 'qvolume']].\
        describe().to_string().replace(" ", "").replace("\n", "") 
    expected_output = '''
                            trd_exctn_dt        prclean       qvolume
    count                         826209  823602.000000  8.262090e+05
    mean   2005-06-30 21:53:21.233102336      99.671788  4.308005e+06
    min              2005-01-03 00:00:00       0.000100  1.000000e+04
    25%              2005-03-31 00:00:00      97.064999  5.000000e+04
    50%              2005-06-28 00:00:00     100.384701  2.000000e+05
    75%              2005-09-30 00:00:00     104.790972  1.795000e+06
    max              2005-12-30 00:00:00    1116.249977  1.263632e+10
    std                              NaN      12.156334  4.755138e+07
    '''

    assert (output == expected_output.replace(" ", "").replace("\n", "")) and \
        (output_shape == expected_shape)


def test_load_mmn_corrected_bond_functionality():
    """Test the functionality of MMN corrected monthly bond data loaded from Open Source Bond Asset Pricing."""

    # Test if the function returns a pandas DataFrame
    assert isinstance(df_mmn, pd.DataFrame)

    # Test if the DataFrame has the expected columns
    expected_columns = ['date', 'cusip', 'bond_ret', 'exretn', 'rating',
       'bond_yield', 'bond_amount_out', 'offering_amt', 'bondprc', 
       'tmt', 'BOND_RET', 'ILLIQ', 'n_trades_month', 'BONDPRC', 'PRFULL',
       'bond_value', 'BOND_VALUE']
    assert all(col in df_mmn.columns for col in expected_columns)

    # Test if the function raises an error when given an invalid data directory
    with pytest.raises(FileNotFoundError):
        load_opensource.load_mmn_corrected_bond(data_dir="invalid_directory")



def test_load_mmn_corrected_bond_validity(df_mmn):
    """Test the validity of MMN corrected monthly bond data loaded from Open Source Bond Asset Pricing."""
    
    # Test if the default date range has the expected start date and end date
    assert df_mmn['date'].min() == '2002-08-31'
    assert df_mmn['date'].max() >= '2022-09-01'
    
    df_mmn_ = df_mmn.copy()
    df_mmn_sample = df_mmn_[(df_mmn_['date'] >= '2005-01-01') & (df_mmn_['date'] <= '2005-12-31')]
    
    output_shape = df_mmn_sample.shape

    output = df_mmn_sample[['cusip', 'bond_ret', 'ILLIQ', 'rating']].\
        describe().to_string().replace(" ", "").replace("\n", "") 
    expected_output = '''
            bond_ret         ILLIQ        rating
    count  27465.000000  30230.000000  35046.000000
    mean       0.001324      1.012048     10.029019
    std        0.029798     32.160654      4.358154
    min       -0.660869   -180.920575      1.000000
    25%       -0.007914      0.015155      6.000000
    50%        0.002896      0.100724      9.000000
    75%        0.012192      0.414362     14.000000
    max        0.430446   5048.979951     22.000000
    '''
    
    expected_shape = (35046, 34)
    assert (output == expected_output.replace(" ", "").replace("\n", "")) and \
        (output_shape == expected_shape)