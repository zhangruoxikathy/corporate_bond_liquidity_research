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


def plot_illiquidity(illiquidity_df, summary_df, title):
    """Plot monthly illiquidity per bond and average & median illiquidity by year.
    
    In case of outliers, a reasonable illquidity range has been chosen to display 
    another zoomed-in plot without outliers.

    Parameters:
        illiquidity_df (pandas.DataFrame): Monthly illiquidity per bond dataframe.
        summary_df (pandas.DataFrame): Summary stats for illiquidity per year. 
        title (str): Desired plot title.

    Returns:
        A plot with two subplots: one displaying all data and the other displaying
        zoomed-in illiquidity without outliers.
    """

    # Prepare time series for plot
    illiquidity_df = illiquidity_df.dropna(subset=['illiq'])
    if 'date' in list(illiquidity_df.columns):
        illiquidity_df['date'] = pd.to_datetime(illiquidity_df['date'])
        illiquidity_df['month'] = illiquidity_df['date'].dt.month
    else:
        illiquidity_df['month_year'] = pd.to_datetime(illiquidity_df['month_year'], format='%Y-%m')
        illiquidity_df['month'] = illiquidity_df['month_year'].dt.month
    
    illiquidity_df['yearmonth'] = illiquidity_df['year'].astype(
            str) + '-' + illiquidity_df['month'].astype(str).str.zfill(2)
    illiquidity_df['yearmonth'] = pd.to_datetime(illiquidity_df['yearmonth'], format='%Y-%m')
    summary_df['year'] = pd.to_datetime(summary_df['year'], format='%Y')
    
    # Create a figure with two subplots arranged vertically
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=100)

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

    illiq_daily_paper = pd.read_csv(OUTPUT_DIR / "illiq_daily_paper.csv")
    illiq_daily_summary_paper = pd.read_csv(OUTPUT_DIR / "illiq_daily_summary_paper.csv")
    mmn_paper = pd.read_csv(OUTPUT_DIR / "mmn_paper.csv")
    illiq_daily_summary_mmn_paper = pd.read_csv(OUTPUT_DIR / "illiq_daily_summary_mmn_paper.csv")
    illiq_daily_new = pd.read_csv(OUTPUT_DIR / "illiq_daily_new.csv")
    illiq_daily_summary_new = pd.read_csv(OUTPUT_DIR / "illiq_daily_summary_new.csv")
    mmn_new = pd.read_csv(OUTPUT_DIR / "mmn_new.csv")
    illiq_daily_summary_mmn_new = pd.read_csv(OUTPUT_DIR / "illiq_daily_summary_mmn_new.csv")
    
    plot_illiquidity(illiq_daily_paper, illiq_daily_summary_paper, "2003-2009")
    plot_illiquidity(mmn_paper, illiq_daily_summary_mmn_paper, "MMN_Corrected, 2003-2009")
    plot_illiquidity(illiq_daily_new, illiq_daily_summary_new, "2003-2023")
    plot_illiquidity(mmn_new, illiq_daily_summary_mmn_new, "MMN_Corrected, 2003-2023")



if __name__ == "__main__":
    main()
