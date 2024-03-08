[Replication of Tables in _The Illiquidity of Corporate Bonds_, Bao, Pan, and Wang (2010).](https://github.com/zhangruoxikathy/corporate_bond_liquidity_research/blob/main/The%20Journal%20of%20Finance%20-%202011%20-%20BAO%20-%20The%20Illiquidity%20of%20Corporate%20Bonds.pdf) 
==================

# About this project

We want to leverage Python to replicate Table 1 and Table 2 from the academic research paper _The Illiquidity of Corporate Bonds_, Bao, Pan, and Wang (2010). Run doit in the terminal for this project will produce tables, summary statistics, 

# Data used: 

1) WRDS BondRet dataset: a cleaned database incorporating two feeds: FINRA’s TRACE (Trade Reporting and Compliance Engine) data for bond transactions, and Mergent FISD data for bond issue and issuer characteristics, reported on a monthly basis

2) Daily TRACE panel data: Maintained by a group of contributors from Open Source Bond Asset Pricing (https://openbondassetpricing.com/), this data includes bond individual level price-relevant data based on FINRA’s TRACE data, reported on a daily basis

3) FINRA’s TRACE data: the original raw data containing individual level bond characteristics, reported on a trade-by-trade basis

4) MMN-corrected WRDS TRACE data: The bond-level panel with characteristics adjusted for market microstructure noise, pulled directly from Open Source Bond Asset Pricing, reported on a monthly basis


# Report:

`Final_Project_Report.tex` and its pdflatex compiled version `Final_Project_Report.pdf` in the `reports` folder contains the high-level overview and replication results of our project.


# General Directory Structure

- `src`: Contains all of the source codes, including load py files, function py files, jupyter notebooks, unit test files. Our replication code follows a general structure of data_processing, table1, and table2. Both table 1 and table 2 codes can run independently, with dependencies on load data files and data_processing. Notebooks are used to walk through our functions and codes. 

- `data`: `pulled` and `manual` folders to store data. In our case, all data are loaded into `pulled` folder but can also be maunually downloaded from wrds or Open Source Bond Asset Pricing (https://openbondassetpricing.com/).

- `output`: Contains replicated tables, summary stats, plots, table latex files.

- `assets`: Contains the original paper, screenshots to input into final latex report.

- We use `doit` Python module as a task runner. It works like `make` and the associated `Makefile`s. To rerun the code, install `doit` (https://pydoit.org/) and execute the command `doit` from the `src` directory. Note that doit is very flexible and can be used to run code commands from the command prompt, thus making it suitable for projects that use scripts written in multiple different programming languages.

- The `.env` file as a container for absolute paths that are private to each collaborator in the project. You can also use it for private credentials, if needed. It should not be tracked in Git.


# Quick Start

`conda` allows for a lot of flexibility, but can often be slow. `pip`, however, is fast for what it does.  You can install the requirements for this project using the `requirements.txt` file specified here. Do this with the following command:
```
conda create -n illiq_p python=3.9
```
Navigate to the this repo, and then use:

```
conda activate illiq_p
pip install -r requirements.txt
```

The requirements file can be created like this:
```
pip list --format=freeze
pip freeze > requirements.txt
```

Now, navigate to the main directory
```
doit
```
That should be it! 

Warning: In the pull_data action, if it freezes for too long, it may be wrds asking for your username and password (but doit might suppress the output), so please enter your username until wrds server information pops up.

And then, use the following commend to run unit test:
```
pytest (or python -m pytest)
```
