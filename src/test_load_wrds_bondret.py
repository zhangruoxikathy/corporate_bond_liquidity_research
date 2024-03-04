'''
Overview
-------------
This Python script contains the unit tests for data extracted from wrds bond return database
via load_wrds_bondret.py.


Requirements
-------------

../data/pulled/Bondret.parquet resulting from load_wrds_bondret.py
../src/load_wrds_bondret.py

'''

import pandas as pd
import numpy as np
import pytest
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import load_wrds_bondret

df_bondret = load_wrds_bondret.load_bondret(data_dir=DATA_DIR)


def test_load_daily_bond_functionality():
    """Test the functionality of monthly bond data loaded from WRDS."""

    # Test if the function returns a pandas DataFrame
    assert isinstance(df_bondret, pd.DataFrame)

    # Test if the DataFrame has the expected columns
    expected_columns = ['cusip', 'date', 'issue_id', 'bond_type', 't_spread',
      'offering_date', 'offering_amt', 'offering_price', 'principal_amt',
      'maturity', 'treasury_maturity', 'coupon', 'day_count_basis', 'dated_date',
      'ncoups', 'amount_outstanding', 'n_mr', 'tmt']
    assert all(col in df_bondret.columns for col in expected_columns)

    # Test if the function raises an error when given an invalid data directory
    with pytest.raises(FileNotFoundError):
        load_wrds_bondret.load_bondret(data_dir="invalid_directory")


def test_load_daily_bond_data_validity():
    """Test the validity of daily bond data loaded from Open Source Bond Asset Pricing."""

    # Test if the default date range has the expected start date and end date
    assert df_bondret['date'].min() == pd.Timestamp('2002-07-31')
    assert df_bondret['date'].max() >= pd.Timestamp('2023-12-31')

    # Test one year of summary statisticas of daily bond data
    df_bondret_sample = df_bondret[df_bondret['year'] == 2005]
    
    output_shape = df_bondret_sample.shape
    expected_shape = (74828, 30)

    output = df_bondret_sample[['cusip', 'offering_amt', 'offering_price', 'n_mr']].\
        describe().to_string().replace(" ", "").replace("\n", "") 
    expected_output = '''
        offering_amt  offering_price          n_mr
    count  7.482800e+04    54860.000000  71496.000000
    mean   3.681025e+05       99.277911      8.776897
    std    4.161223e+05        5.206366      4.131448
    min    1.500000e+01        8.845000      1.000000
    25%    1.500000e+05       99.460000      6.000000
    50%    2.500000e+05       99.748000      8.000000
    75%    4.500000e+05       99.950000     11.000000
    max    7.790000e+06      120.000000     21.000000
    '''

    assert (output == expected_output.replace(" ", "").replace("\n", "")) and \
        (output_shape == expected_shape)
