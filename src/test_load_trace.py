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
import datetime
import pytest
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import load_intraday

START_DATE = '01-01-2003'
END_DATE = '12-31-2023'
df_intraday = load_intraday.load_intraday_TRACE(START_DATE, END_DATE, data_dir=DATA_DIR)

def test_load_trace():
    """Test the functionality of daily bond data loaded from WRDS TRACE."""

    expected_columns = ['cusip_id', 'trd_exctn_dt', 'trd_exctn_tm', 'days_to_sttl_ct',
                        'lckd_in_ind', 'wis_fl', 'msg_seq_nb', 'entrd_vol_qt', 'rptd_pr',
                        'orig_msg_seq_nb']

    assert expected_columns == list(df_intraday.columns)
    assert df_intraday['trd_exctn_dt'].min() == datetime.date(2003, 1, 2)
    assert df_intraday['trd_exctn_dt'].max() >= datetime.date(2023, 6, 30)

    df_sample = df_intraday[
          (df_intraday['trd_exctn_dt'] >= pd.Timestamp('01-01-2005').date())
        & (df_intraday['trd_exctn_dt'] <= pd.Timestamp('12-31-2005').date())
    ]

    assert df_sample.shape == (3553636, 10)

    expected_describe = '''
                 rptd_pr
    count  3.553636e+06
    mean   9.700144e+01
    std    1.327767e+01
    min   -1.227000e+00
    25%    9.575000e+01
    50%    9.981900e+01
    75%    1.026500e+02
    max    1.117875e+03
    '''.replace(" ", "").replace("\n", "")

    result_describe = df_sample[['rptd_pr']].describe().to_string().replace(" ", "").replace("\n", "")
    assert (result_describe == expected_describe)


test_load_trace()