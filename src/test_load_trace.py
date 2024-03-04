'''
Overview
-------------
This Python script contains the unit tests for data extracted from wrds TRACE database
via load_intraday.py.


Requirements
-------------

../data/pulled/IntradayTRACE.parquet resulting from load_intraday.py
../src/load_intraday.py

'''

import pandas as pd
import numpy as np
import pytest
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import load_intraday

START_DATE = '01-01-2003'
END_DATE = '12-31-2023'
df_intraday = load_intraday.load_intraday_TRACE(START_DATE, END_DATE, data_dir=DATA_DIR)

def test_load_trace():
    """ Test """

    expected_columns = [
        'cusip_id', 'bond_sym_id', 'trd_exctn_dt', 'trd_exctn_tm', 'days_to_sttl_ct', 'lckd_in_ind', 'wis_fl',
        'sale_cndtn_cd', 'msg_seq_nb', 'trc_st', 'trd_rpt_dt', 'trd_rpt_tm', 'entrd_vol_qt', 'rptd_pr', 'yld_pt',
        'asof_cd', 'orig_msg_seq_nb', 'rpt_side_cd', 'cntra_mp_id'
    ]

    assert expected_columns == list(df_intraday.columns)
    assert df['trd_exctn_dt'].min() == pd.Timestamp('2003-01-02').date()
    assert df['trd_exctn_dt'].max() == pd.Timestamp('2023-06-30').date()

    df_sample = df[
          (df['trd_exctn_dt'] >= pd.Timestamp('01-01-2005').date())
        & (df['trd_exctn_dt'] <= pd.Timestamp('12-31-2005').date())
    ]

    assert df_sample.shape == (3553636, 19)

    expected_describe = '''
               entrd_vol_qt       rptd_pr        yld_pt
    count  3.553636e+06  3.553636e+06  3.507542e+06
    mean   7.930956e+05  9.700144e+01  1.042461e+01
    std    3.000620e+07  1.327767e+01  4.643079e+02
    min    1.000000e-02 -1.227000e+00  1.000000e-06
    25%    1.000000e+04  9.575000e+01  4.497000e+00
    50%    2.500000e+04  9.981900e+01  5.133938e+00
    75%    1.000000e+05  1.026500e+02  8.105000e+00
    max    1.000000e+10  1.117875e+03  2.244510e+05
    '''.replace(" ", "").replace("\n", "")

    result_describe = df_sample[['entrd_vol_qt', 'rptd_pr', 'yld_pt']] \
                        .describe().to_string().replace(" ", "").replace("\n", "")
    assert (result_describe == expected_describe)
