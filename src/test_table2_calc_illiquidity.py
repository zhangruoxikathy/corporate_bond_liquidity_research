import pandas as pd
import numpy as np
import config

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

import table2_calc_illiquilidy


cleaned_df = table2_calc_illiquilidy.clean_merged_data('2003-04-14', '2009-06-30')



def test_clean_merged_data():
    output = cleaned_df[['trd_exctn_dt', 'prclean', 'n']].describe().to_string().replace(" ", "").replace("\n", "") 
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
    df = table2_calc_illiquilidy.calc_deltaprc(cleaned_df)
    output = df[['prclean', 'deltap', 'deltap_lag']].describe().to_string().replace(" ", "").replace("\n", "") 

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



def test_calc_annual_illiquidity_table_daily():
    df = table2_calc_illiquilidy.calc_deltaprc(cleaned_df)
    illiq_daily, table2_daily = table2_calc_illiquilidy.calc_annual_illiquidity_table_daily(df)

    # Key trends in mean illiq
    mean_illiq_trend = (
        table2_daily.loc[table2_daily['Year'] == 2005, 'Mean illiq'].values[0] < 0.9 * table2_daily.loc[table2_daily['Year'] == 2003, 'Mean illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2006, 'Mean illiq'].values[0] < 0.9 * table2_daily.loc[table2_daily['Year'] == 2005, 'Mean illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2007, 'Mean illiq'].values[0] > 1.3 * table2_daily.loc[table2_daily['Year'] == 2006, 'Mean illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2008, 'Mean illiq'].values[0] > 3 * table2_daily.loc[table2_daily['Year'] == 2003, 'Mean illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2008, 'Mean illiq'].values[0] > 1.3 * table2_daily.loc[table2_daily['Year'] == 2007, 'Mean illiq'].values[0]
    )

    # First increasing and then decreasing trends in median illiq， since median better captures the trend compared to the mean being less affected by
    # positive outliers
    decreasing_trend = (
        table2_daily.loc[table2_daily['Year'] == 2004, 'Median illiq'].values[0] < table2_daily.loc[table2_daily['Year']== 2003, 'Median illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2005, 'Median illiq'].values[0] < table2_daily.loc[table2_daily['Year'] == 2004, 'Median illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2006, 'Median illiq'].values[0] < table2_daily.loc[table2_daily['Year'] == 2005, 'Median illiq'].values[0]
    )

    increasing_trend = (
        table2_daily.loc[table2_daily['Year'] == 2007, 'Median illiq'].values[0] > table2_daily.loc[table2_daily['Year'] == 2006, 'Median illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2008, 'Median illiq'].values[0] > table2_daily.loc[table2_daily['Year'] == 2007, 'Median illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 2009, 'Median illiq'].values[0] > table2_daily.loc[table2_daily['Year'] == 2008, 'Median illiq'].values[0]
    )

    median_illiq_trend = decreasing_trend and increasing_trend
    
    full_trend = (
        table2_daily.loc[table2_daily['Year'] == 'Full', 'Mean illiq'].values[0] > table2_daily.loc[table2_daily['Year'] == 2007, 'Mean illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 'Full', 'Mean illiq'].values[0] < table2_daily.loc[table2_daily['Year'] == 2008, 'Mean illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 'Full', 'Median illiq'].values[0] > table2_daily.loc[table2_daily['Year'] == 2007, 'Median illiq'].values[0] and
        table2_daily.loc[table2_daily['Year'] == 'Full', 'Robust t stat'].values[0] > 10
    )
    
    assert (mean_illiq_trend and median_illiq_trend and full_trend)
    
    

def test_calc_annual_illiquidity_table_spd():
    df = table2_calc_illiquilidy.calc_deltaprc(cleaned_df)
    illiq_daily, table2_daily = table2_calc_illiquilidy.calc_annual_illiquidity_table_daily(df)
    table2_spd = table2_calc_illiquilidy.calc_annual_illiquidity_table_spd(df)

    # Both mean and median follow first increasing and then decreasing trends
    mean_decreasing_trend = (
        table2_spd.loc[table2_spd['Year'] == 2004, 'Mean implied gamma'].values[0] < table2_spd.loc[table2_spd['Year'] == 2003, 'Mean implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2005, 'Mean implied gamma'].values[0] < table2_spd.loc[table2_spd['Year'] == 2004, 'Mean implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2006, 'Mean implied gamma'].values[0] < table2_spd.loc[table2_spd['Year'] == 2005, 'Mean implied gamma'].values[0]
    )

    mean_increasing_trend = (
        table2_spd.loc[table2_spd['Year'] == 2007, 'Mean implied gamma'].values[0] > table2_spd.loc[table2_spd['Year'] == 2006, 'Mean implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2008, 'Mean implied gamma'].values[0] > table2_spd.loc[table2_spd['Year'] == 2007, 'Mean implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2009, 'Mean implied gamma'].values[0] > table2_spd.loc[table2_spd['Year'] == 2008, 'Mean implied gamma'].values[0]
    )
    
    median_decreasing_trend = (
        table2_spd.loc[table2_spd['Year'] == 2004, 'Median implied gamma'].values[0] < table2_spd.loc[table2_spd['Year'] == 2003, 'Median implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2005, 'Median implied gamma'].values[0] < table2_spd.loc[table2_spd['Year'] == 2004, 'Median implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2006, 'Median implied gamma'].values[0] < table2_spd.loc[table2_spd['Year'] == 2005, 'Median implied gamma'].values[0]
    )

    median_increasing_trend = (
        table2_spd.loc[table2_spd['Year'] == 2007, 'Median implied gamma'].values[0] > table2_spd.loc[table2_spd['Year'] == 2006, 'Median implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2008, 'Median implied gamma'].values[0] > table2_spd.loc[table2_spd['Year'] == 2007, 'Median implied gamma'].values[0] and
        table2_spd.loc[table2_spd['Year'] == 2009, 'Median implied gamma'].values[0] > table2_spd.loc[table2_spd['Year'] == 2008, 'Median implied gamma'].values[0]
    )
    
    median_trend = median_decreasing_trend and median_increasing_trend
    median_illiq_trend = mean_decreasing_trend and mean_increasing_trend
    
    # All means are slightly higher than median, indicates positively skewed data for all years
    all_means_higher = (table2_spd['Mean implied gamma'] > table2_spd['Median implied gamma']).all()

    # Spread implied γ are more than one order of magnitude smaller than the empirically observed γ for individual bonds.
    spdilliq_lower_than_dailyilliq = (table2_spd['Mean implied gamma'] < 0.1 * table2_daily['Mean illiq']).all()
    
    assert (median_trend and median_illiq_trend and all_means_higher and spdilliq_lower_than_dailyilliq)