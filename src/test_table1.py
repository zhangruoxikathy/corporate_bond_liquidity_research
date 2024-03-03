'''
this file designs the unit test for table 1 
since the exact same processing is applied to both sample and all bonds reported,
the below unit tests only include those for the sample.
Generally, the unit test is designed in a way to compare the numbers from our output 
with the original paper, allowing a certain percentage of tolerance and a limit number of data outlier exceeding the tolerance

see the below the benchmark criteria for different variables for our table 1 data to pass unit tests: 

1) # of bonds: +- 15% of original paper
2) Issuance: +-15% of orginal paper, allowing 1 outlier
3) Rating: +-15% of original paper, allowing 1 outlier
4) Maturity: +-20% of original paper, allowing 1 outlier
5) Coupon: +-20% of original paper, allowing 1 outlier
6) Age: +-20% of original paper, allowing 1 outlier
7) Turnover: +-100% of original paper, allowing 6 outlier
8) Trd Size: +-50% of original paper, allowing 3 outlier
9) #Trades: +-100% of original paper, allowing 2 outlier
10) Avg Ret: trend in mean and median of average return data is consistent between output and paper, 
    with differences in each year having a standard deviation < 0.1
11) Volatility: +-30% of original paper, allowing 3 outlier
12) Price: +-20% of original paper, allowing 3 outlier

From the benchmark criteria, 
we get fairly close data for # of bonds, Issuance, Rating, Maturity, Coupon, Age, Trd Size, Price, and Volatility

we get no so close data for Turnover, presumably because the monthly trading volume, the numerator of the metric, 
is directly from the Bondret dataset; the original paper could use a different calculation method for a bond's montly trading volume
We also get no so close data for Trd Size and # of trades, since the intra-day trade data are from the intermediary steps of Alex Dickerson's code,
combining with the filters in our data prcoessing steps (essentially a combination from original TRACE, the TRACE cleaned Daily data output, and Monthly Bondret)
Such integration can introduce variations, justifying the observed differences in the data.

we get very similar trend for Avg Return, though the eaxct number are somewhat different

'''



import pandas as pd
import numpy as np
import config
import table1

from table1 import df_sample_result

from pathlib import Path


pd.set_option('display.max_columns', None)


output_data = df_sample_result

output_data = output_data.T


def test_total_bond_number(output_data):
    '''
    test if total bond number is within +- 15% of the total bond number appeared in the paper 
    '''
    total_bond_number = output_data['cusip_count']

    tolerance_percent = 0.15
    error_count = 0

    paper_total_bond_number = {
        2003: 744,
        2004: 951,
        2005: 911,
        2006: 748,
        2007: 632,
        2008: 501,
        2009: 373
    }

    for year, expected_count in paper_total_bond_number.items():
        actual_count = total_bond_number.loc[year]
        lower_bound = expected_count * (1 - tolerance_percent)  
        upper_bound = expected_count * (1 + tolerance_percent)  
        assert lower_bound <= actual_count <= upper_bound, f"Total bond number for {year} is {actual_count}, which is not within "f"{tolerance*100}% of the expected {expected_count}"


test_total_bond_number(output_data)


def test_issuance_data(output_data):
    '''
    test if issuance data summary stats is within +-15% of that in the paper, allowing 1 outlier
    '''

    issuance = output_data[['issuance_avg', 'issuance_median', 'issuance_std']]

    paper_issuance = {
        2003: {'issuance_avg': 1013, 'issuance_median': 987, 'issuance_std': 735},
        2004: {'issuance_avg': 930, 'issuance_median': 750, 'issuance_std': 714},
        2005: {'issuance_avg': 990, 'issuance_median': 771, 'issuance_std': 696},
        2006: {'issuance_avg': 983, 'issuance_median': 797, 'issuance_std': 659},
        2007: {'issuance_avg': 1001, 'issuance_median': 798, 'issuance_std': 676},
        2008: {'issuance_avg': 1032, 'issuance_median': 848, 'issuance_std': 705},
        2009: {'issuance_avg': 1071, 'issuance_median': 990, 'issuance_std': 726},
    }
    tolerance_percent = 0.15 
    error_count = 0

    for year, metrics in paper_issuance.items():
        for metric, expected_value in metrics.items():
            actual_value = issuance.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance_percent)
            upper_bound = expected_value * (1 + tolerance_percent)
            if not (lower_bound <= actual_value <= upper_bound):
                error_count += 1
                if error_count > 1:
                    raise AssertionError(
                        f"More than 1 'Issuance' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )


test_issuance_data(output_data)

def test_rating_data(output_data):

    '''
    test if rating summary stats is within +-15% of that in the paper, allowing 1 outlier
    '''

    ratings = output_data[['n_mr_avg', 'n_mr_median', 'n_mr_std']]
    
    paper_ratings = {
        2003: {'n_mr_avg': 5.36, 'n_mr_median': 5.22, 'n_mr_std': 2.13},
        2004: {'n_mr_avg': 5.55, 'n_mr_median': 5.08, 'n_mr_std': 2.32},
        2005: {'n_mr_avg': 5.67, 'n_mr_median': 5.00, 'n_mr_std': 2.40},
        2006: {'n_mr_avg': 5.38, 'n_mr_median': 5.00, 'n_mr_std': 2.30},
        2007: {'n_mr_avg': 5.33, 'n_mr_median': 5.00, 'n_mr_std': 2.35},
        2008: {'n_mr_avg': 5.71, 'n_mr_median': 5.92, 'n_mr_std': 2.35},
        2009: {'n_mr_avg': 6.60, 'n_mr_median': 6.67, 'n_mr_std': 2.13},
    }
    tolerance = 0.15
    error_count = 0

    for year, metrics in paper_ratings.items():
        for metric, expected_value in metrics.items():
            actual_value = ratings.loc[year, metric] 
            lower_bound = expected_value * (1-tolerance) 
            upper_bound = expected_value * (1+tolerance) 
            if not lower_bound <= actual_value <= upper_bound:
                error_count += 1
                if error_count > 1:
                    raise AssertionError(
                        f"More than 1 'Rating' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )


test_rating_data(output_data)


def test_maturity_data(output_data):

    '''
    test if maturity data summary stats is within +-20% of that in the paper, allowing 1 outlier
    '''

    maturities = output_data[['tmt_avg','tmt_median','tmt_std']]
   
    paper_maturities = {
        2003: {'tmt_avg': 7.38, 'tmt_median': 5.21, 'tmt_std': 6.87},
        2004: {'tmt_avg': 7.68, 'tmt_median': 5.16, 'tmt_std': 7.28},
        2005: {'tmt_avg': 7.19, 'tmt_median': 4.62, 'tmt_std': 7.31},
        2006: {'tmt_avg': 6.54, 'tmt_median': 4.36, 'tmt_std': 6.98},
        2007: {'tmt_avg': 6.25, 'tmt_median': 3.75, 'tmt_std': 7.06},
        2008: {'tmt_avg': 5.55, 'tmt_median': 3.75, 'tmt_std': 7.05},
        2009: {'tmt_avg': 5.80, 'tmt_median': 3.66, 'tmt_std': 7.37},
    }

    tolerance = 0.2
    error_count = 0

    for year, metrics in paper_maturities.items():
        for metric, expected_value in metrics.items():
            actual_value = maturities.loc[year, metric]  
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            if not lower_bound <= actual_value <= upper_bound:
                error_count += 1
                if error_count > 1:
                    raise AssertionError(
                        f"More than 1 'Maturity' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )


test_maturity_data(output_data)

def test_coupon_data(output_data):

    '''
    test if coupon data summary stats is within +-20% of that in the paper, allowing 1 outlier
    '''

    coupon = output_data[['coupon_y_avg','coupon_y_median', 'coupon_y_std']]

    paper_coupons = {
        2003: {'coupon_y_avg': 5.84, 'coupon_y_median': 6.00, 'coupon_y_std': 1.63},
        2004: {'coupon_y_avg': 5.71, 'coupon_y_median': 6.00, 'coupon_y_std': 1.69},
        2005: {'coupon_y_avg': 5.63, 'coupon_y_median': 5.80, 'coupon_y_std': 1.67},
        2006: {'coupon_y_avg': 5.50, 'coupon_y_median': 5.50, 'coupon_y_std': 1.65},
        2007: {'coupon_y_avg': 5.47, 'coupon_y_median': 5.62, 'coupon_y_std': 1.65},
        2008: {'coupon_y_avg': 5.55, 'coupon_y_median': 5.70, 'coupon_y_std': 1.65},
        2009: {'coupon_y_avg': 5.80, 'coupon_y_median': 5.88, 'coupon_y_std': 1.60},
    }

    tolerance = 0.2
    coupon_error_count = 0

    for year, metrics in paper_coupons.items():
        for metric, expected_value in metrics.items():
            actual_value = coupon.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            if not lower_bound <= actual_value <= upper_bound:
                coupon_error_count += 1
                if coupon_error_count > 1:
                    raise AssertionError(
                        f"More than 1 'Coupon' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )

test_coupon_data(output_data)

def test_age_data(output_data):
    '''
    test if age data summary stats is within +-20% of that in the paper, allowing 1 outlier
    '''

    paper_ages = {
        2003: {'age_avg': 2.73, 'age_median': 1.94, 'age_std': 2.68},
        2004: {'age_avg': 3.21, 'age_median': 2.41, 'age_std': 2.91},
        2005: {'age_avg': 3.93, 'age_median': 3.25, 'age_std': 2.90},
        2006: {'age_avg': 4.52, 'age_median': 3.87, 'age_std': 2.71},
        2007: {'age_avg': 5.46, 'age_median': 4.61, 'age_std': 2.83},
        2008: {'age_avg': 6.42, 'age_median': 5.66, 'age_std': 2.93},
        2009: {'age_avg': 7.23, 'age_median': 6.50, 'age_std': 3.03},
    }

    ages = output_data[['age_avg','age_median', 'age_std']]
    tolerance = 0.2
    age_error_count = 0

    for year, metrics in paper_ages.items():
        for metric, expected_value in metrics.items():
            actual_value = ages.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            if not lower_bound <= actual_value <= upper_bound:
                age_error_count += 1
                if age_error_count > 1:
                    raise AssertionError(
                        f"More than 1 'Age' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )


test_age_data(output_data)

def test_turnover_data(output_data):
    '''
    test if turnover data summary stats is within +-100% of that in the paper, allowing 6 outlier
    '''

    paper_turnover = {
        2003: {'turnover_avg': 11.83, 'turnover_median': 8.52, 'turnover_std': 9.83},
        2004: {'turnover_avg': 9.47, 'turnover_median': 7.09, 'turnover_std': 7.71},
        2005: {'turnover_avg': 7.51, 'turnover_median': 5.92, 'turnover_std': 5.87},
        2006: {'turnover_avg': 5.83, 'turnover_median': 4.99, 'turnover_std': 3.99},
        2007: {'turnover_avg': 4.87, 'turnover_median': 4.11, 'turnover_std': 3.26},
        2008: {'turnover_avg': 4.70, 'turnover_median': 4.19, 'turnover_std': 2.83},
        2009: {'turnover_avg': 5.98, 'turnover_median': 5.06, 'turnover_std': 4.12},
    }

    turnover_stats = output_data[['turnover_avg', 'turnover_median', 'turnover_std']]
    tolerance = 1
    turnover_error_count = 0

    for year, metrics in paper_turnover.items():
        for metric, expected_value in metrics.items():
            actual_value = turnover_stats.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            if not lower_bound <= actual_value <= upper_bound:
                turnover_error_count += 1
                if turnover_error_count > 6:
                    raise AssertionError(
                        f"More than 6 'Turnover' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )
    
test_turnover_data(output_data)

def test_num_trade_data(output_data):
    '''
    test if trade data summary stats is within +-100% of that in the paper, allowing 2 outlier
    '''
    paper_trades = {
        2003: {'#trade_avg': 248, '#trade_median': 153, '#trade_std': 372},
        2004: {'#trade_avg': 187, '#trade_median': 127, '#trade_std': 201},
        2005: {'#trade_avg': 209, '#trade_median': 121, '#trade_std': 316},
        2006: {'#trade_avg': 151, '#trade_median': 110, '#trade_std': 121},
        2007: {'#trade_avg': 148, '#trade_median': 107, '#trade_std': 129},
        2008: {'#trade_avg': 219, '#trade_median': 144, '#trade_std': 219},
        2009: {'#trade_avg': 408, '#trade_median': 221, '#trade_std': 511},
    }

    trades = output_data[['#trade_avg','#trade_median', '#trade_std']]
    tolerance = 1
    trade_error_count = 0

    for year, metrics in paper_trades.items():
        for metric, expected_value in metrics.items():
            actual_value = trades.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            if not lower_bound <= actual_value <= upper_bound:
                trade_error_count += 1
                if trade_error_count > 2:
                    raise AssertionError(
                        f"More than 3 '#Trades' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )

test_num_trade_data(output_data)


def test_trade_size_data(output_data):
    '''
    Test if trade size data summary stats is within +-50% of that in the paper, allowing 3 outlier
    '''
    paper_trade_sizes = {
        2003: {'trade_size_avg': 585, 'trade_size_median': 462, 'trade_size_std': 469},
        2004: {'trade_size_avg': 557, 'trade_size_median': 415, 'trade_size_std': 507},
        2005: {'trade_size_avg': 444, 'trade_size_median': 331, 'trade_size_std': 412},
        2006: {'trade_size_avg': 409, 'trade_size_median': 306, 'trade_size_std': 366},
        2007: {'trade_size_avg': 356, 'trade_size_median': 267, 'trade_size_std': 335},
        2008: {'trade_size_avg': 248, 'trade_size_median': 180, 'trade_size_std': 240},
        2009: {'trade_size_avg': 206, 'trade_size_median': 134, 'trade_size_std': 217},
    }

    trade_sizes = output_data[['trade_size_avg', 'trade_size_median', 'trade_size_std']]
    tolerance = 0.5
    trade_size_error_count = 0

    for year, metrics in paper_trade_sizes.items():
        for metric, expected_value in metrics.items():
            actual_value = trade_sizes.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            if not lower_bound <= actual_value <= upper_bound:
                trade_size_error_count += 1
                if trade_size_error_count > 3:
                    raise AssertionError(
                        f"More than 1 'Trade Size' values are out of the acceptable range. "
                        f"Year {year}, metric '{metric}' has a value of {actual_value}, "
                        f"which is not within {tolerance*100}% of the expected {expected_value}."
                    )

test_trade_size_data(output_data)


def test_avg_return_data(output_data):
    '''
    test if the trend in mean and median of average return data is consistent between output and paper
    '''

    # Values from the paper for each year
    paper_data = {
        2003: {'Avg_return_avg': 0.52, 'Avg_return_median': 0.36},
        2004: {'Avg_return_avg': 0.40, 'Avg_return_median': 0.30},
        2005: {'Avg_return_avg': 0.00, 'Avg_return_median': 0.16},
        2006: {'Avg_return_avg': 0.37, 'Avg_return_median': 0.29},
        2007: {'Avg_return_avg': 0.44, 'Avg_return_median': 0.46},
        2008: {'Avg_return_avg': -0.40, 'Avg_return_median': 0.36},
        2009: {'Avg_return_avg': 1.07, 'Avg_return_median': 0.80},
    }

    differences_mean = []   # measures the differences of Avg_return_avg between output and paper
    differences_median = [] # measures the differences of Avg_return_median between output and paper 

    for year in paper_data.keys():
        paper_mean = paper_data[year]['Avg_return_avg']
        paper_median = paper_data[year]['Avg_return_median']
        output_mean = output_data.loc[year, 'Avg_return_avg']
        output_median = output_data.loc[year, 'Avg_return_median']
        
        differences_mean.append((paper_mean - output_mean))
        differences_median.append((paper_median - output_median))

    std_dev_mean = np.std(differences_mean)
    std_dev_median = np.std(differences_median)
    
    # if the standard deviation in either mean difference or median differernce is above 0.1, then we assume the trend is not consistant

    if std_dev_mean >= 0.1 or std_dev_median >= 0.1:
        raise AssertionError(
            "the trend between Average return in output and in paper is not consistent"
        )

test_avg_return_data(output_data)

def test_volatility_data(output_data):
    '''
    test if volatility data summary stats is within +-30% of that in the paper, allowing 3 outlier
    '''

    paper_volatility = {
    2003: {'volatility_avg': 2.49, 'volatility_median': 2.25, 'volatility_std': 1.48},
    2004: {'volatility_avg': 1.72, 'volatility_median': 1.59, 'volatility_std': 0.98},
    2005: {'volatility_avg': 1.62, 'volatility_median': 1.24, 'volatility_std': 1.39},
    2006: {'volatility_avg': 1.01, 'volatility_median': 0.87, 'volatility_std': 1.18},
    2007: {'volatility_avg': 1.39, 'volatility_median': 1.08, 'volatility_std': 1.07},
    2008: {'volatility_avg': 5.61, 'volatility_median': 3.14, 'volatility_std': 8.22},
    2009: {'volatility_avg': 4.94, 'volatility_median': 3.09, 'volatility_std': 5.11},
    }

    volatility_stats = output_data[['volatility_avg','volatility_median', 'volatility_std']]
    
    tolerance = 0.3
    volatility_error_count = 0

    for year, metrics in paper_volatility.items():
        for metric, expected_value in metrics.items():
            actual_value = volatility_stats.loc[year, metric]
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            
            if not lower_bound <= actual_value <= upper_bound:
                volatility_error_count += 1
                
                if volatility_error_count > 3:
                    raise AssertionError(
                        f"More than 1 'Volatility' values are out of the acceptable range for year {year}. "
                        f"Metric '{metric}' has a value of {actual_value}, which is not within "
                        f"{tolerance*100}% of the expected {expected_value}."
                    )
                    

test_volatility_data(output_data)


def test_price_data(output_data):
    '''
    test if price data summary stats is within +-20% of that in the paper, allowing 3 outlier
    '''

    paper_price = {
        2003: {'price_avg': 108, 'price_median': 109, 'price_std': 9},
        2004: {'price_avg': 106, 'price_median': 103, 'price_std': 9},
        2005: {'price_avg': 104, 'price_median': 103, 'price_std': 9},
        2006: {'price_avg': 102, 'price_median': 101, 'price_std': 9},
        2007: {'price_avg': 103, 'price_median': 101, 'price_std': 12},
        2008: {'price_avg': 99, 'price_median': 102, 'price_std': 16},
        2009: {'price_avg': 102, 'price_median': 102, 'price_std': 13},
    }

    price_stats = output_data[['prclean_avg','prclean_median', 'prclean_std']]
    
    tolerance = 0.2
    price_error_count = 0

    for year, metrics in paper_price.items():
        for metric, expected_value in metrics.items():
           
            actual_value = price_stats.loc[year, metric.replace('price_', 'prclean_')]
            
            lower_bound = expected_value * (1 - tolerance)
            upper_bound = expected_value * (1 + tolerance)
            
            if not lower_bound <= actual_value <= upper_bound:
                price_error_count += 1
                
                if price_error_count > 3:
                    raise AssertionError(
                        f"More than 1 'Price' values are out of the acceptable range for year {year}. "
                        f"Metric '{metric}' has a value of {actual_value}, which is not within "
                        f"{tolerance*100}% of the expected {expected_value}."
                    )
                    

test_price_data(output_data)