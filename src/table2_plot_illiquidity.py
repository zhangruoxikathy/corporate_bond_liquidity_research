import load_fred
import config
from pathlib import Path
DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


import table2_calc_illiquidity as calc_illiquidity
# sns.set()

# df = load_fred.load_fred(data_dir=DATA_DIR)

# (
#     100 * 
#     df[['CPIAUCNS', 'GDPC1']]
#     .rename(columns={'CPIAUCNS':'Inflation', 'GDPC1':'Real GDP'})
#     .dropna()
#     .pct_change(4)
#     ).plot()
# plt.title("Inflation and Real GDP, Seasonally Adjusted")
# plt.ylabel('Percent change from 12-months prior')
# filename = OUTPUT_DIR / 'example_plot.png'
# plt.savefig(filename);


def plot_illiquidity(illiquidity_df, summary_df, title):

    # Prepare time series for plot
    illiquidity_df = illiquidity_df.dropna(subset=['illiq'])

    illiquidity_df['month'] = illiquidity_df['month_year'].dt.month
    illiquidity_df['yearmonth'] = illiquidity_df['year'].astype(
        str) + '-' + illiquidity_df['month'].astype(str).str.zfill(2)
    illiquidity_df['yearmonth'] = pd.to_datetime(illiquidity_df['yearmonth'], format='%Y-%m')
    summary_df['year'] = pd.to_datetime(summary_df['year'], format='%Y')
    
    # Create a figure with two subplots arranged vertically
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Scatter plot of illiquidity by year, all data
    sns.scatterplot(data=illiquidity_df, x='yearmonth', y='illiq', alpha=0.3, ax=axs[0])

    # Line plot for the mean illiquidity by year
    sns.lineplot(data=summary_df, x='year', y='mean illiq', color='red', lw=2, ax=axs[0])
    sns.lineplot(data=summary_df, x='year', y='median', color='purple', lw=2, ax=axs[0])

    axs[0].set_title(f'Illiquidity by Year with Mean Illiquidity, {title}')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Illiquidity')
    axs[0].grid(True)

    # Scatter plot of illiquidity by year, zoomed in
    sns.scatterplot(data=illiquidity_df[(illiquidity_df['illiq'] <= 800) &
                                        (illiquidity_df['illiq'] >= -500)],
                    x='yearmonth', y='illiq', alpha=0.3, ax=axs[1])

    # Line plot for the mean illiquidity by year
    sns.lineplot(data=summary_df, x='year', y='mean illiq', color='red', lw=2, ax=axs[1])
    sns.lineplot(data=summary_df, x='year', y='median', color='purple', lw=2, ax=axs[1])

    axs[1].set_title(f'Illiquidity by Year with Mean Illiquidity, Zoomed In, {title}')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Illiquidity')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/f"illiq_plot_{title}.png")
    plt.show()
    

def main():
    
    # Define dates
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = '2003-04-14'
    end_date = '2009-06-30' 
    
    # Replicate table 2 in the paper
    cleaned_df_paper = calc_illiquidity.clean_merged_data(start_date, end_date)
    df_paper = calc_illiquidity.calc_deltaprc(cleaned_df_paper)
    illiq_daily_paper, table2_daily_paper = calc_illiquidity.calc_annual_illiquidity_table_daily(df_paper)
    illiq_daily_summary_paper = calc_illiquidity.create_summary_stats(illiq_daily_paper)
    
    plot_illiquidity(illiq_daily_paper, illiq_daily_summary_paper, "2003-2009")

    # Using MMN corrected data
    mmn_paper, table2_daily_mmn_paper = calc_illiquidity.calc_illiq_w_mmn_corrected(start_date, end_date,
                                                                   cleaned_df_paper)
    illiq_daily_summary_mmn_paper = calc_illiquidity.create_summary_stats(mmn_paper)
    
    plot_illiquidity(mmn_paper, illiq_daily_summary_mmn_paper, "MMN_Corrected, 2003-2009")
    
    
    # Update table to the present
    cleaned_df_new = calc_illiquidity.clean_merged_data('2003-04-14', today)
    df_new = calc_illiquidity.calc_deltaprc(cleaned_df_new)

    illiq_daily_new, table2_daily_new = calc_illiquidity.calc_annual_illiquidity_table_daily(df_new)
    illiq_daily_summary_new = calc_illiquidity.create_summary_stats(illiq_daily_new)

    plot_illiquidity(illiq_daily_new, illiq_daily_summary_new, "2003-2023")
    
    # Using MMN corrected data
    mmn_new, table2_daily_mmn_new = calc_illiquidity.calc_illiq_w_mmn_corrected(start_date, end_date,
                                                               cleaned_df_new)
    illiq_daily_summary_mmn_new = calc_illiquidity.create_summary_stats(mmn_new)
    
    plot_illiquidity(mmn_new, illiq_daily_summary_mmn_new, "MMN_Corrected, 2003-2023")


if __name__ == "__main__":
    main()
