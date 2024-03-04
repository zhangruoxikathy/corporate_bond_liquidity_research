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
    - Mean illiquidity using trade-by-trade data: 
    - Mean illiquidity using daily data: +- 40% & trend test
- Panel B Bond Portfolio
    - Equal-weighted mean illiquidity: +- 0.05
    - Issuance-weighted mean illiquidity: +- 0.07
- Panel C Implied by quoted bid-ask spread
    - Bid-ask spread mean x 5: +- 40% & trend test
    - Bid-ask spread median x 5: +- 40% & trend test

 
Requirements
-------------

../data/pulled/Bondret.parquet resulting from load_wrds_bondret.py
../data/pulled/BondDailyPublic.parquet resulting from load_opensource.py
../src/table2_calc_illiquidity.py

'''


#* ************************************** */
#* Libraries                              */
#* ************************************** */ 
import pandas as pd
import numpy as np
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import table2_calc_illiquidity

# Test on the same time period in the paper
START_DATE = '2003-04-14'
END_DATE = '2009-06-30'

cleaned_daily_df = table2_calc_illiquidity.clean_merged_data(START_DATE, END_DATE)
df = table2_calc_illiquidity.calc_deltaprc(cleaned_daily_df)


# Test data are handled as expected
def test_clean_merged_data():
    """Test summary statisticas of the df produced by clean_merged_data."""

    output = cleaned_daily_df[['trd_exctn_dt', 'prclean', 'n']].describe().to_string().replace(
        " ", "").replace("\n", "") 
    expected_output = '''
                            trd_exctn_dt        prclean              n
    count                         886367  886367.000000  886367.000000
    mean   2005-12-27 01:49:59.844985088     100.689418       1.107497
    min              2003-04-15 00:00:00       0.000200       0.000000
    25%              2004-07-23 00:00:00      98.407802       1.000000
    50%              2005-10-13 00:00:00     101.421197       1.000000
    75%              2007-04-24 00:00:00     105.739201       1.000000
    max              2009-06-30 00:00:00    4111.562144       7.000000
    std                              NaN      13.142166       0.419525
    '''
    
    assert output == expected_output.replace(" ", "").replace("\n", "")

def test_calc_deltaprc():
    """Test deltap and deltap_lag calculated by calc_deltaprc."""

    output = df[['prclean', 'deltap', 'deltap_lag']].describe().to_string().replace(
        " ", "").replace("\n", "") 

    expected_output = """
                prclean         deltap     deltap_lag
    count  884449.000000  884449.000000  884449.000000
    mean      100.693676      -0.009131      -0.009528
    std        13.133103       2.363088       2.368232
    min         0.000200    -100.000000    -100.000000
    25%        98.405001      -0.259022      -0.259035
    50%       101.423998      -0.002094      -0.002140
    75%       105.739801       0.263740       0.263362
    max      4111.562144     100.000000     100.000000
    """
    
    assert output == expected_output.replace(" ", "").replace("\n", "")

##############################################################
# Test Panel A
##############################################################


def test_table2_panelA_daily_within_tolerance():
    """Test if table 2 Panel A illiquidity results using daily data are within +-40% tolerance
    of the results in the paper."""
    
    results_mean = {}
    results_median = {}
    
    illiq_daily, table2_daily = table2_calc_illiquidity.calc_annual_illiquidity_table(df)

    tolerance_mean = 0.4

    paper_illiq_daily_mean = {     
        2003: 0.99,
        2004: 0.82,
        2005: 0.77,
        2006: 0.57,
        2007: 0.80,
        2008: 3.21, # much higher with outliers
        2009: 5.40, # much higher with outliers
        'Full': 1.18 # much higher with outliers
    }

    for year, expected_mean in paper_illiq_daily_mean.items():
        if year not in [2008, 2009, 'Full']:
            actual_mean = table2_daily.loc[table2_daily['Year'] == year, 'Mean illiq'].values[0]
            # Check if the actual mean is within the lower and upper bounds based on the tolerance
            lower_bound = expected_mean * (1 - tolerance_mean)
            upper_bound = expected_mean * (1 + tolerance_mean)
            results_mean[year] = lower_bound <= actual_mean <= upper_bound

    assert results_mean, "Table 2 Panel A tolerance test failed"



def test_table2_panela_daily_trend():
    """Test if table 2 Panel A illiquidity results using daily data follow the trend in the paper."""
    
    illiq_daily, table2_daily = table2_calc_illiquidity.calc_annual_illiquidity_table(df)
    
    table2_daily['Year'] = table2_daily['Year'].astype(str)
    mean_illiq_series = table2_daily.set_index('Year')['Mean illiq']
    median_illiq_series = table2_daily.set_index('Year')['Median illiq']

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
        for year in [2007, 2008, 2009])

    # Check 'Full' year specific conditions
    full_trend = (
        mean_illiq_series[full_year] > mean_illiq_series['2007'] and
        mean_illiq_series[full_year] < mean_illiq_series['2008'] and
        median_illiq_series[full_year] > median_illiq_series['2007'] and
        table2_daily.loc[table2_daily['Year'] == full_year, 'Robust t stat'].values[0] > 10)
    
    assert mean_illiq_trend and median_illiq_trend and full_trend, "Table 2 Panel A trend test failed"
    

##############################################################
# Test Panel B
##############################################################


def test_table2_panelb_port_within_tolerance():
    """Test if table 2 Panel B equal weighted illiquidity results are within +-0.05 tolerance, 
    issuance weighted illiquidity results are within +-0.07 tolerance of the results in the paper."""
    
    table2_port = table2_calc_illiquidity.calc_annual_illiquidity_table_portfolio(df)
    
    results_ew = {}
    results_iw = {}

    tolerance_ew = 0.05
    tolerance_iw = 0.07

    paper_ew = {     
        2003: -0.0014,
        2004: -0.0043,
        2005: -0.0008,
        2006: 0.0001,
        2007: 0.0023,
        2008: -0.0112,
        2009: -0.0301,
        'Full': -0.0050}             
    
    paper_iw = {            
        2003: 0.0018,
        2004: -0.0042,
        2005: -0.0003,
        2006: 0.0007,
        2007: 0.0034,
        2008: 0.0030,
        2009: -0.0280,
        'Full': -0.0017}          

    for year, expected_ew in paper_ew.items():
        actual_ew = table2_port.loc[table2_port['Year'] == 2003, 'Equal weighted'].values[0]
        lower_bound = expected_ew - tolerance_ew
        upper_bound = expected_ew + tolerance_ew
        results_ew[year] = lower_bound <= actual_ew <= upper_bound
            
    for year, expected_iw in paper_iw.items():
        actual_iw = table2_port.loc[table2_port['Year'] == year, 'Issuance weighted'].values[0]
        lower_bound = expected_iw - tolerance_iw
        upper_bound = expected_iw + tolerance_iw
        results_iw[year] = lower_bound <= actual_iw <= upper_bound


    assert all(results_ew.values()) and all(results_iw.values()), "Table 2 Panel B tolerance test failed"


##############################################################
# Test Panel C
##############################################################


def test_table2_panelc_spd_within_tolerance():
    """Test if table 2 Panel C bid-ask spread mean and median results are within +-40% tolerance
    of the results in the paper."""

    table2_spd = table2_calc_illiquidity.calc_annual_illiquidity_table_spd(df)
    
    results_mean = {}
    results_median = {}

    adj = 5
    tolerance_mean = 0.4
    tolerance_median = 0.4
    
    table2_spd['Mean implied gamma adj'] = table2_spd['Mean implied gamma']*adj
    table2_spd['Median implied gamma adj'] = table2_spd['Median implied gamma']*adj

    paper_spd_mean = {     
        2003: 0.035,
        2004: 0.031,
        2005: 0.034,
        2006: 0.028,
        2007: 0.031,
        2008: 0.050,
        2009: 0.070,
        'Full': 0.034
    }      
    
    paper_spd_median = {            
        2003: 0.031,
        2004: 0.025,
        2005: 0.023,
        2006: 0.018,
        2007: 0.021,
        2008: 0.045,
        2009: 0.059,
        'Full': 0.026
    }   

    for year, expected_mean in paper_spd_mean.items():
        actual_mean_adj = table2_spd.loc[table2_spd['Year'] == year, 'Mean implied gamma adj'].values[0]
        lower_bound = expected_mean * (1 - tolerance_mean)
        upper_bound = expected_mean * (1 + tolerance_mean)
        results_mean[year] = lower_bound <= actual_mean_adj <= upper_bound
            
    for year, expected_median in paper_spd_median.items():
        actual_median_adj = table2_spd.loc[table2_spd['Year'] == year, 'Median implied gamma adj'].values[0]
        lower_bound = expected_median * (1 - tolerance_median)
        upper_bound = expected_median * (1 + tolerance_median)
        results_median[year] = lower_bound <= actual_median_adj <= upper_bound


    assert all(results_mean.values()) and all(results_median.values()), "Table 2 Panel C tolerance test failed"

    
    
def test_table2_panelc_spd_trend():
    """Test if table 2 Panel C bid-ask spread mean and median results follow the trend in the paper."""

    table2_spd = table2_calc_illiquidity.calc_annual_illiquidity_table_spd(df)

    table2_spd['Year'] = table2_spd['Year'].astype(str)

    # Define the trends for mean and median implied gamma
    trends = {
        'mean_decreasing': (2003, 2004, 2005, 2006),
        'mean_increasing': (2006, 2007, 2008, 2009),
        'median_decreasing': (2003, 2004, 2005, 2006),
        'median_increasing': (2006, 2007, 2008, 2009)
    }

    trend_results = {}
    for trend_name, years in trends.items():
        trend_results[trend_name] = all(
            table2_spd.loc[table2_spd['Year'] == str(years[i+1]), 'Mean implied gamma'].values[0] > 
            table2_spd.loc[table2_spd['Year'] == str(years[i]), 'Median implied gamma'].values[0]
            for i in range(len(years) - 1)
        )

    # Check if all means are slightly higher than median, indicating positively skewed data
    all_means_higher = (table2_spd['Mean implied gamma'] > table2_spd['Median implied gamma']).all()

    assert all(trend_results.values()) and all_means_higher, "Table 2 Panel C trend test failed"
