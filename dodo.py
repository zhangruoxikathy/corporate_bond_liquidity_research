"""
Dependency Steps In Order:

 - config file to establish required directories
 - wrds monthly data file
 - open source daily data file

 - trade-by-trade TRACE data file


Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based
"""


import sys
sys.path.insert(1, './src/')

import config
from pathlib import Path
from doit.tools import run_once
import platform

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)



# fmt: off
################################################################
## Helper functions for automatic execution of Jupyter notebooks
################################################################
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def check_file_exists(targets):
    """Returns True if all specified target files exist."""
    return all(Path(target).exists() for target in targets)


def get_os():
    os_name = platform.system()
    if os_name == "Windows":
        return "windows"
    elif os_name == "Darwin":
        return "nix"
    elif os_name == "Linux":
        return "nix"
    else:
        return "unknown"
os_type = get_os()


def copy_notebook_to_folder(notebook_stem, origin_folder, destination_folder):
    origin_path = Path(origin_folder) / f"{notebook_stem}.ipynb"
    destination_path = Path(destination_folder) / f"_{notebook_stem}.ipynb"
    if os_type == "nix":
        command =  f"cp {origin_path} {destination_path}"
    else:
        command = f"copy  {origin_path} {destination_path}"
    return command


########
## Tasks
########



def task_pull_data():
    file_dep = [
        "./src/config.py",
        "./src/load_wrds_bondret.py",
        "./src/load_opensource.py",
    ]
    targets = [
        Path(DATA_DIR) / "pulled" / file for file in
        [
            ## src/load_wrds_bondret.py
            "Bondret.parquet",
            ## src/load_opensource.py
            "BondDailyPublic.parquet",
            "WRDS_MMN_Corrected_Data.parquet",
            ## src/load_intraday.py
            # "IntradayTRACE.parquet",
        ]
    ]

    actions = [
        "ipython src/config.py",
        "ipython src/load_wrds_bondret.py",
        "ipython src/load_opensource.py",
    ]
    return {
        "actions": actions,
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
        "verbosity": 2, # Print everything immediately. This is important in
        # case WRDS asks for credentials.
        "uptodate": [check_file_exists(targets)]
    }




def task_pull_intraday_data():
    file_dep = [
        "./src/config.py",
        "./src/load_wrds_bondret.py",
        "./src/load_opensource.py",
        "./src/intraday_TRACE_Pull.py",
        "./src/load_intraday.py",
        "./data/pulled/Bondret.parquet",
        "./data/pulled/BondDailyPublic.parquet"
        ]
    targets = [
        Path(DATA_DIR) / "pulled" / file for file in
        [
            "IntradayTRACE.parquet",
        ]
    ]

    actions = [
        "ipython src/load_intraday.py"
    ]
    return {
        "actions": actions,
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
        "verbosity": 2, # Print everything immediately. This is important in
        # case WRDS asks for credentials.
        "uptodate": [check_file_exists(targets)]
    }




def task_summary_data():
    actions = [
        "ipython src/table1.py",
        "ipython src/table2_calc_illiquidity.py"
    ]
    targets = [OUTPUT_DIR / fname for fname in [
        "table2_panelA_trade_by_trade_paper.csv",
        "illiq_daily_paper.csv",
        "illiq_summary_paper.csv",
        "table2_panelA_daily_paper.csv",
        "table2_panelB_paper.csv",
        "table2_panelC_paper.csv",
        "mmn_paper.csv",
        "illiq_daily_summary_mmn_paper.csv",
        "table2_daily_mmn_paper.csv",
        "table2_panelA_trade_by_trade_new.csv",
        "illiq_daily_new.csv",
        "illiq_summary_new.csv",
        "table2_panelA_daily_new.csv",
        "table2_panelB_new.csv",
        "table2_panelC_new.csv",
        "mmn_new.csv",
        "illiq_daily_summary_mmn_new.csv",
        "table2_daily_mmn_new.csv",
        "table1_panelA.csv",
        "table1_panelB.csv",
        "table1_panelB_uptodate.csv",
        "table1_panelA_uptodate.csv"
    ]]


    file_dep = [
        "./src/table1.py",
        "./src/table2_calc_illiquidity.py"
    ]

    file_dep.extend(
        Path(DATA_DIR) / "pulled" / file for file in
        [
            "Bondret.parquet",
            "BondDailyPublic.parquet",
            "WRDS_MMN_Corrected_Data.parquet",
            "IntradayTRACE.parquet",
        ]
    )
    return {
        'actions': actions,
        'file_dep': file_dep,
        #'task_dep': [task_pull_data],
        'targets': targets,
    }


def task_generate_plots():

    file_dep = [OUTPUT_DIR / fname for fname in [
        "illiq_daily_paper.csv",
        "illiq_summary_paper.csv",
        "mmn_paper.csv",
        "illiq_daily_summary_mmn_paper.csv",
        "illiq_daily_new.csv",
        "illiq_summary_new.csv",
        "mmn_new.csv",
        "illiq_daily_summary_mmn_new.csv"
    ]]

    file_dep.extend(['./src/table2_plot_illiquidity.py'])

    targets = [OUTPUT_DIR / fname for fname in [
        'illiq_plot_2003-2009.png',
        'illiq_plot_2003-2023.png',
        'illiq_plot_MMN_Corrected, 2003-2009.png',
        'illiq_plot_MMN_Corrected, 2003-2023.png'
    ]]

    actions = [
        'ipython src/table2_plot_illiquidity.py'
    ]

    return {
        'actions': actions,
        'file_dep': file_dep,
        #'task_dep': [task_summary_data],
        'targets': targets,
    }


def task_produce_latex_tables():
    file_dep = [OUTPUT_DIR / fname for fname in [
        "table2_panelA_trade_by_trade_paper.csv",
        "illiq_daily_paper.csv",
        "illiq_summary_paper.csv",
        "table2_panelA_daily_paper.csv",
        "table2_panelB_paper.csv",
        "table2_panelC_paper.csv",
        "mmn_paper.csv",
        "illiq_daily_summary_mmn_paper.csv",
        "table2_daily_mmn_paper.csv",
        "table2_panelA_trade_by_trade_new.csv",
        "illiq_daily_new.csv",
        "illiq_summary_new.csv",
        "table2_panelA_daily_new.csv",
        "table2_panelB_new.csv",
        "table2_panelC_new.csv",
        "mmn_new.csv",
        "illiq_daily_summary_mmn_new.csv",
        "table2_daily_mmn_new.csv",
        "table1_panelA.csv",
        "table1_panelB.csv",
        "table1_panelB_uptodate.csv",
        "table1_panelA_uptodate.csv",
        "table1_panelA.csv",
        "table1_panelB.csv"
    ]]

    file_dep.extend([
        './src/table1_pandas_to_latex.py',
        './src/table2_pandas_to_latex.py',
    ])

    targets = [OUTPUT_DIR / fname for fname in [
        'illiq_summary_paper.tex',
        'illiq_daily_summary_mmn_paper.tex',
        'table2_panelA_trade_by_trade_paper.tex',
        'table2_panelA_daily_paper.tex',
        'table2_panelB_paper.tex',
        'table2_panelC_paper.tex',
        'table2_daily_mmn_paper.tex',
        'illiq_summary_new.tex',
        'illiq_daily_summary_mmn_new.tex',
        'table2_panelA_trade_by_trade_new.tex',
        'table2_panelA_daily_new.tex',
        'table2_panelB_new.tex',
        'table2_panelC_new.tex',
        'table2_daily_mmn_new.tex',
        'table1_panelA.tex',
        'table1_panelB.tex',
        'table1_panelA_uptodate.tex',
        'table1_panelB_uptodate.tex'
    ]]

    actions = [
        'ipython src/table1_pandas_to_latex.py',
        'ipython src/table2_pandas_to_latex.py',
    ]

    return {
        'actions': actions,
        'file_dep': file_dep,
        #'task_dep': [task_summary_data],
        'targets': targets,
    }


def task_compile_latex_report():
    file_dep = [OUTPUT_DIR / fname for fname in [
        'illiq_summary_paper.tex',
        'illiq_daily_summary_mmn_paper.tex',
        'table2_panelA_trade_by_trade_paper.tex',
        'table2_panelA_daily_paper.tex',
        'table2_panelB_paper.tex',
        'table2_panelC_paper.tex',
        'table2_daily_mmn_paper.tex',
        'illiq_summary_new.tex',
        'illiq_daily_summary_mmn_new.tex',
        'table2_panelA_trade_by_trade_new.tex',
        'table2_panelA_daily_new.tex',
        'table2_panelB_new.tex',
        'table2_panelC_new.tex',
        'table2_daily_mmn_new.tex',
        'table1_panelA.tex',
        'table1_panelB.tex',
        'table1_panelA_uptodate.tex',
        'table1_panelB_uptodate.tex',
        'illiq_plot_2003-2009.png',
        'illiq_plot_2003-2023.png',
        'illiq_plot_MMN_Corrected, 2003-2009.png',
        'illiq_plot_MMN_Corrected, 2003-2023.png'
    ]]

    file_dep.extend([
        './assets/table1_screenshot.jpg',
        './assets/table2_screenshot.jpg'
    ])

    targets = [f'./reports/Final_Project_Report.pdf']

    return {
        'actions': [
            'cd reports && pdflatex Final_Project_Report.tex',
            'cd ..'
        ],
        'targets': targets,
        'file_dep': file_dep,
        #"task_dep": [task_generate_plots, task_produce_latex_tables],
        "clean": True,
    }


def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    notebooks = [
        "DataProcessing.ipynb",
        "summary_statistics.ipynb"
        "table1.ipynb",
        "table2_part1.ipynb",
        "table2_part2.ipynb"
    ]

    stems = [notebook.split(".")[0] for notebook in notebooks]

    file_dep = [
        # 'load_other_data.py',
        # *[Path(OUTPUT_DIR) / f"_{stem}.py" for stem in stems],
        "./data/pulled/Bondret.parquet",
        ## src/load_opensource.py
        "./data/pulled/BondDailyPublic.parquet",
        "./data/pulled/WRDS_MMN_Corrected_Data.parquet",
         ## src/load_intraday.py
        "./data/pulled/IntradayTRACE.parquet"]


    targets = [
        ## Notebooks converted to HTML
        *[OUTPUT_DIR / f"{stem}.html" for stem in stems],
    ]

    actions = [
        *[jupyter_execute_notebook(notebook) for notebook in stems],
        *[jupyter_to_html(notebook) for notebook in stems],

        # *[copy_notebook_to_folder(notebook, Path("./src"), OUTPUT_DIR) for notebook in stems],
        # *[copy_notebook_to_folder(notebook, Path("./src"), "./docs") for notebook in stems],
        # *[jupyter_clear_output(notebook) for notebook in stems]
        # *[jupyter_to_python(notebook, build_dir) for notebook in notebooks_to_run],
    ]
    return {
        "actions": actions,
        # "targets": targets,
        # "task_dep": [task_pull_data],
        "file_dep": file_dep,
        "clean": True,
    }

