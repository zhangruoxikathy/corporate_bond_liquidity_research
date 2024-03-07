'''
Overview
-------------
This Python script designs the unit test for table 2. Since based on table 1,
our data have discrepancies with that used in the paper, the below unit tests
is designed in a way to compare trends in illiquidity and sometimes with a percentage
or absolute value of tolerance accepted. Moreover, the first two tests are conducted to
ensure the data are loaded and cleaned as expected.

Here is a list of tests done on table 2, including:

- Panel A Individual Bonds (The mean and average monthly illiquidity per bond per year)
    - Mean illiquidity using trade-by-trade data: +- 40% & trend test


Requirements
-------------

../data/pulled/IntradayTRACE.parquet resulting from load_intraday.py
../src/table2_calc_illiquidity.py

'''

# * ************************************** */
# * Libraries                              */
# * ************************************** */
import pandas as pd
import numpy as np
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import table2_calc_illiquidity

# Test on the same time period in the paper
START_DATE = '2003-04-14'
END_DATE = '2009-06-30'

cleaned_intraday_df = table2_calc_illiquidity.clean_intraday(START_DATE, END_DATE)
df_deltapr = table2_calc_illiquidity.calc_deltaprc(cleaned_intraday_df)


def test_clean_intraday():
    """Test summary statisticas of the df produced by clean_intraday."""
    output = cleaned_intraday_df[['trd_exctn_dt', 'prclean']].describe().to_string().replace(" ", "").replace("\n",
                                                                                                              "")
    expected_output = '''
                            trd_exctn_dt       prclean
    count                       13276314  1.327631e+07
    mean   2006-03-18 06:36:35.777774080  9.743166e+01
    min              2003-04-14 00:00:00  5.005000e+00
    25%              2004-08-11 00:00:00  9.625000e+01
    50%              2005-12-02 00:00:00  1.000000e+02
    75%              2007-12-05 00:00:00  1.037630e+02
    max              2009-06-30 00:00:00  9.999900e+02
    std                              NaN  1.397011e+01
    '''

    assert output == expected_output.replace(" ", "").replace("\n", "")


def test_calc_intraday_deltaprc():
    """Test deltap and deltap_lag calculated by calc_deltaprc."""

    output = df_deltapr[['prclean', 'deltap', 'deltap_lag']].describe().to_string().replace(
        " ", "").replace("\n", "")

    expected_output = """
                prclean        deltap    deltap_lag
    count  1.327439e+07  1.327439e+07  1.327439e+07
    mean   9.743152e+01 -9.018127e-04 -9.038407e-04
    std    1.396972e+01  1.844254e+00  1.844417e+00
    min    5.005000e+00 -1.000000e+02 -1.000000e+02
    25%    9.625000e+01 -1.645609e-01 -1.645386e-01
    50%    1.000000e+02  0.000000e+00  0.000000e+00
    75%    1.037630e+02  1.813643e-01  1.813442e-01
    max    9.999900e+02  1.000000e+02  1.000000e+02
    """

    assert output == expected_output.replace(" ", "").replace("\n", "")


##############################################################
# Test Panel A
##############################################################


def test_table2_panelA_intraday_within_tolerance():
    """Test if table 2 Panel A illiquidity results using intrday data are within +-40% tolerance
    of the results in the paper."""

    results_mean = {}
    results_median = {}

    _, table2_tbt = table2_calc_illiquidity.calc_annual_illiquidity_table(df_deltapr)

    tolerance_mean = 0.4

    paper_illiq_tbt_mean = {
        2003: 0.64,     # much higher with outliers
        2004: 0.60,     # much higher with outliers
        2005: 0.52,
        2006: 0.40,
        2007: 0.44,
        2008: 1.02,     # much higher with outliers
        2009: 1.35,     # much higher with outliers
        'Full': 0.63    # much higher with outliers
    }

    for year, expected_mean in paper_illiq_tbt_mean.items():
        if year not in [2003, 2004, 2008, 2009, 'Full']:
            actual_mean = table2_tbt.loc[table2_tbt['Year'] == year, 'Mean illiq'].values[0]
            # Check if the actual mean is within the lower and upper bounds based on the tolerance
            lower_bound = expected_mean * (1 - tolerance_mean)
            upper_bound = expected_mean * (1 + tolerance_mean)
            results_mean[year] = lower_bound <= actual_mean <= upper_bound

    assert results_mean, "Table 2 Panel A Trade-by-Trade tolerance test failed"


def test_table2_panelA_intraday_trend():
    """Test if table 2 Panel A illiquidity results using intraday data follow the trend in the paper."""

    _, table2_tbt = table2_calc_illiquidity.calc_annual_illiquidity_table(df_deltapr)

    table2_tbt['Year'] = table2_tbt['Year'].astype(str)
    mean_illiq_series = table2_tbt.set_index('Year')['Mean illiq']
    median_illiq_series = table2_tbt.set_index('Year')['Median illiq']

    years = mean_illiq_series.index[:-1]  # Exclude'Full'
    full_year = 'Full'

    # Check mean and median illiq trend
    mean_illiq_trend = all(
        mean_illiq_series[str(year)] > mean_illiq_series[str(year + 1)]
        for year in [2004, 2005]) and all(
        mean_illiq_series[str(year)] > mean_illiq_series[str(year - 1)]
        for year in [2007, 2008, 2009])

    median_illiq_trend = all(
        median_illiq_series[str(year)] > median_illiq_series[str(year + 1)]
        for year in [2003, 2004, 2005]) and all(
        median_illiq_series[str(year)] > median_illiq_series[str(year - 1)]
        for year in [2008, 2009])

    # Check 'Full' year specific conditions
    full_trend = (
            mean_illiq_series[full_year] > mean_illiq_series['2007'] and
            mean_illiq_series[full_year] < mean_illiq_series['2008'] and
            median_illiq_series[full_year] > median_illiq_series['2007'] and
            table2_tbt.loc[table2_tbt['Year'] == full_year, 'Robust t stat'].values[0] > 2)

    assert mean_illiq_trend and median_illiq_trend and full_trend, "Table 2 Panel A Trade-by-Trade trend test failed"


test_clean_intraday()
test_calc_intraday_deltaprc()
test_table2_panelA_intraday_within_tolerance()
test_table2_panelA_intraday_trend()
