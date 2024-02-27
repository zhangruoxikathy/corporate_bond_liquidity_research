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

import misc_tools
import load_wrds_bondret
import load_opensource
import data_processing as data
import calc_illiquilidy as illiq




cleaned_df = illiq.clean_merged_data('2003-04-14', '2009-06-30')
df = illiq.calc_deltaprc(cleaned_df)
illiq_daily, table2_daily = illiq.calc_annual_illiquidity_table_daily(df)
table2_spd = illiq.calc_annual_illiquidity_table_spd(df) 


# Set display, but doesn't affect formatting to LaTeX
pd.set_option('display.float_format', lambda x: '%.4f' % x)
# Sets format for printing to LaTeX
float_format_func = lambda x: '{:.4f}'.format(x)

# Produce latex tables
illiq_daily_summary = illiq_daily.describe()
latex_illiq_daily = illiq_daily_summary.to_latex(index=False, float_format=float_format_func, column_format='rrr', escape=False)
latex_table2_daily = table2_daily.to_latex(index=False, float_format=float_format_func, column_format='l|rrrr', escape=False)
latex_table2_spd = table2_spd.to_latex(index=False, float_format=float_format_func, column_format='l|rr', escape=False)

# print(latex_illiq_daily)
# print(latex_table2_daily)
# print(latex_table2_spd)


# LaTeX document content
latex_document = f"""
\\documentclass{{article}}
\\usepackage{{booktabs}}

\\begin{{document}}

\\section*{{Summary Statistics: Monthly Illiquidity Using Daily Data}}
{latex_illiq_daily}

\\section*{{Panel A: Individual Bonds, Daily Data}}
{latex_table2_daily}

\\section*{{Panel C: Implied by Quoted Bid-Ask Spreads}}
{latex_table2_spd}

\\end{{document}}
"""


path = OUTPUT_DIR / f'pandas_to_latex_table2.tex'
with open(path, "w") as text_file:
    text_file.write(latex_document)

