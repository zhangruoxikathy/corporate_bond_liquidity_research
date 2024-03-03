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
import table2_calc_illiquidity as illiq

pd.set_option('display.float_format', lambda x: '%.4f' % x)


def produce_illiq_latex(summary_string):
    """Load and process per cusip monthly illiquidity summary stats table, output to latex."""

    summary = pd.read_csv(OUTPUT_DIR / f"{summary_string}.csv")

    # Sets format for printing to LaTeX
    float_format_func = lambda x: '{:.4f}'.format(x)
    
    latex_summary = summary.to_latex(index=False, float_format=float_format_func,
                                     column_format='r'*len(summary.columns), escape=False)

    with open(OUTPUT_DIR / f"{summary_string}.tex", "w") as f:
        f.write(latex_summary)
    
    return latex_summary



def produce_table2_latex(panel_string):
    """Load and process panels in table 2, output to latex."""

    panel = pd.read_csv(OUTPUT_DIR / f"{panel_string}.csv").T
    panel.columns = panel.iloc[0]
    panel = panel[1:]

    # Sets format for printing to LaTeX
    float_format_func = lambda x: '{:.4f}'.format(x)
    
    latex_panel = panel.to_latex(index=True, float_format=float_format_func,
                                 column_format='l|'+ 'r'*len(panel.columns), escape=False)
    
    with open(OUTPUT_DIR / f"{panel_string}.tex", "w") as f:
        f.write(latex_panel)
    
    return latex_panel


def main():
    
    # Produce latex tables for time frames in the paper
    latex_illiq_daily_summary_paper = produce_illiq_latex("illiq_summary_paper")
    latex_table2_daily_paper = produce_table2_latex("table2_daily_paper")
    latex_table2_port_paper = produce_table2_latex("table2_port_paper")
    latex_table2_spd_paper = produce_table2_latex("table2_spd_paper")

    # Produce latex tables using up-to-date data
    latex_illiq_daily_summary_new = produce_illiq_latex("illiq_summary_new")
    latex_table2_daily_new = produce_table2_latex("table2_daily_new")
    latex_table2_port_new = produce_table2_latex("table2_port_new")
    latex_table2_spd_new = produce_table2_latex("table2_spd_new")


    # LaTeX document content

    latex_document = f"""
    \\documentclass{{article}}
    \\usepackage{{booktabs}}
    \\usepackage{{geometry}} % For setting margins
    \\geometry{{a4paper, margin=1in}} % Set appropriate margins

    \\begin{{document}}

    \\section*{{Summary Statistics: Monthly Illiquidity Using Daily Data}}
    \\begin{{center}}
    {latex_illiq_daily_summary_paper}
    \\end{{center}}

    \\section*{{Panel A: Individual Bonds, Daily Data}}
    \\begin{{center}}
    {latex_table2_daily_paper}
    \\end{{center}}

    \\section*{{Panel B: Bond Portfolios}}
    \\begin{{center}}
    {latex_table2_port_paper}
    \\end{{center}}

    \\section*{{Panel C: Implied by Quoted Bid-Ask Spreads}}
    \\begin{{center}}
    {latex_table2_spd_paper}
    \\end{{center}}

    \\end{{document}}
    """

    path = OUTPUT_DIR / f'pandas_to_latex_table2.tex'
    with open(path, "w") as text_file:
        text_file.write(latex_document)


if __name__ == "__main__":
    main()
