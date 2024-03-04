r"""
You can test out the latex code in the following minimal working
example document:

\documentclass{article}
\usepackage{booktabs}
\begin{document}
First document. This is a simple example, with no 
extra parameters or packages included.

\begin{table}
\centering
YOUR LATEX TABLE CODE HERE
%\input{example_table.tex}
\end{table}
\end{document}

"""
import pandas as pd
import numpy as np
import config
from pathlib import Path


DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR

df_sample = pd.read_csv(OUTPUT_DIR / "table1_panelA.csv")
df_all = pd.read_csv(OUTPUT_DIR / "table1_panelB.csv")

# replace 'coupon_y' with 'coupon' in Unnamed: 0
# replace 'n_mr' with 'rating' in Unnamed: 0
# replace 'Avg_return' with 'Avf Ret' in Unnamed: 0
# replace 'trade_size' with 'Trd Size' in Unnamed: 0

df_sample['Unnamed: 0'] = df_sample['Unnamed: 0'].str.replace('coupon_y', 'coupon')
df_sample['Unnamed: 0'] = df_sample['Unnamed: 0'].str.replace('n_mr', 'rating')
df_sample['Unnamed: 0'] = df_sample['Unnamed: 0'].str.replace('Avg_return', 'Avf Ret')
df_sample['Unnamed: 0'] = df_sample['Unnamed: 0'].str.replace('trade_size', 'Trd Size')

df_all['Unnamed: 0'] = df_all['Unnamed: 0'].str.replace('coupon_y', 'coupon')
df_all['Unnamed: 0'] = df_all['Unnamed: 0'].str.replace('n_mr', 'rating')
df_all['Unnamed: 0'] = df_all['Unnamed: 0'].str.replace('Avg_return', 'Avf Ret')
df_all['Unnamed: 0'] = df_all['Unnamed: 0'].str.replace('trade_size', 'Trd Size')

float_format_func = lambda x: '{:.2f}'.format(x)


def transform_to_multi_index(df):
    """
    Transforms a given DataFrame with prefixed column names into a multi-index DataFrame.
    
    Parameters:
    - df: DataFrame with columns formatted as 'prefix_subcategory'.
    
    Returns:
    - A multi-index DataFrame with the first level being the prefix and the second level being the subcategories.
    """
    prefixes = set(name.split('_')[0] for name in df['Unnamed: 0'] if '_' in name)

    # Initialize an empty DataFrame to store the multi-indexed columns
    multi_index_df = pd.DataFrame()

    # Loop through each prefix to create multi-index columns
    for prefix in prefixes:
        # Filter columns related to the current prefix
        related_columns = [col for col in df['Unnamed: 0'] if col.startswith(prefix)]
        
        # Extract the data for these columns across all years
        data = df[df['Unnamed: 0'].isin(related_columns)].drop('Unnamed: 0', axis=1).T
        
        # Rename the columns of data to reflect subcategories (e.g., avg, median, std)
        data.columns = [col.split('_')[1] for col in related_columns]
        
        # Add the prefix as the top level of the column index
        data.columns = pd.MultiIndex.from_product([[prefix], data.columns])
        
        # Concatenate this data to the multi_index_df DataFrame
        multi_index_df = pd.concat([multi_index_df, data], axis=1)

    # Reset index to make the years a regular column and transpose for correct orientation
    multi_index_df = multi_index_df.reset_index().rename(columns={'index': 'Year'})
    multi_index_df = multi_index_df.set_index('Year').T

    # Only keep decimals to 2 places
    multi_index_df = multi_index_df.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    return multi_index_df

multi_df_sample = transform_to_multi_index(df_sample)
multi_df_all = transform_to_multi_index(df_all)

latex_format = f'''
\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{pdflscape}}
\\usepackage{{multirow}}

\\begin{{document}}

\\begin{{landscape}}

\\begin{{table}}[ht]
\\centering
\\caption{{Panel A: Bonds in Our Sample}}

{multi_df_sample.to_latex(multirow=True, multicolumn=True, multicolumn_format='c', float_format=float_format_func)}

\\end{{table}}

\\begin{{table}}[ht]
\\centering
\\caption{{Panel B: All Bonds Reported in TRACE}}

{multi_df_all.to_latex(multirow=True, multicolumn=True, multicolumn_format='c', float_format=float_format_func)}

\\end{{table}}

\\end{{landscape}}

\\end{{document}}

'''

with open(OUTPUT_DIR / "table1_pandas_to_latex.tex", "w") as f:
    f.write(latex_format)



