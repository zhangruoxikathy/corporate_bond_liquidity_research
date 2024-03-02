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
import table2_calc_illiquilidy as illiq



illiq_daily_summary_paper = pd.read_csv(OUTPUT_DIR / "illiq_summary_paper.csv", index_col=0)
table2_daily_paper = pd.read_csv(OUTPUT_DIR / "table2_daily_paper.csv").T
table2_port_paper = pd.read_csv(OUTPUT_DIR / "table2_port_paper.csv").T
table2_spd_paper = pd.read_csv(OUTPUT_DIR / "table2_spd_paper.csv").T


# Set display, but doesn't affect formatting to LaTeX
pd.set_option('display.float_format', lambda x: '%.4f' % x)
# Sets format for printing to LaTeX
float_format_func = lambda x: '{:.4f}'.format(x)

# Produce latex tables
latex_illiq_daily_summary_paper = illiq_daily_summary_paper.to_latex(
    index=True, float_format=float_format_func, column_format='rrr', escape=False)
latex_table2_daily_paper = table2_daily_paper.to_latex(
    index=False, float_format=float_format_func, column_format='l|rrrr', escape=False)
latex_table2_port_paper = table2_port_paper.to_latex(
    index=False, float_format=float_format_func, column_format='l|rrrr', escape=False)
latex_table2_spd_paper = table2_spd_paper.to_latex(
    index=False, float_format=float_format_func, column_format='l|rr', escape=False)

# print(latex_illiq_daily)
# print(latex_table2_daily)
# print(latex_table2_spd)


with open("illiq_summary_paper.tex", "w") as f:
    f.write(latex_illiq_daily_summary_paper)

with open("table2_daily_paper.tex", "w") as f:
    f.write(latex_table2_daily_paper)

with open("table2_port_paper.tex", "w") as f:
    f.write(latex_table2_port_paper)

with open("table2_spd_paper.tex", "w") as f:
    f.write(latex_table2_spd_paper)


# LaTeX document content
latex_document = f"""
\\documentclass{{article}}
\\usepackage{{booktabs}}

\\begin{{document}}

\\section*{{Summary Statistics: Monthly Illiquidity Using Daily Data}}
{latex_illiq_daily_summary_paper}

\\section*{{Panel A: Individual Bonds, Daily Data}}
{latex_table2_daily_paper}

\\section*{{Panel B: Bond Portfolios}}
{latex_table2_port_paper}

\\section*{{Panel C: Implied by Quoted Bid-Ask Spreads}}
{latex_table2_spd_paper}

\\end{{document}}
"""


path = OUTPUT_DIR / f'pandas_to_latex_table2.tex'
with open(path, "w") as text_file:
    text_file.write(latex_document)

