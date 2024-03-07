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

To quickest way to run code in this repo is to use the following steps. First, note that you must have TexLive installed on your computer and available in your path.
You can do this by downloading and installing it from here ([windows](https://tug.org/texlive/windows.html#install) and [mac](https://tug.org/mactex/mactex-download.html) installers).
Having installed LaTeX, open a terminal and navigate to the root directory of the project and create a conda environment using the following command:
```
conda create -n blank python=3.12
conda activate blank
```
and then install the dependencies with pip
```
pip install -r requirements.txt
```
You can then navigate to the `src` directory and then run 
```
doit
```

## Other commands

You can run the unit test, including doctests, with the following command:
```
pytest --doctest-modules
```
You can build the documentation with:
```
rm ./src/.pytest_cache/README.md 
jupyter-book build -W ./
```
Use `del` instead of rm on Windows



# Dependencies and Virtual Environments

## Working with `pip` requirements

`conda` allows for a lot of flexibility, but can often be slow. `pip`, however, is fast for what it does.  You can install the requirements for this project using the `requirements.txt` file specified here. Do this with the following command:
```
pip install -r requirements.txt
```

The requirements file can be created like this:
```
pip list --format=freeze
```

## Working with `conda` environments

The dependencies used in this environment (along with many other environments commonly used in data science) are stored in the conda environment called `blank` which is saved in the file called `environment.yml`. To create the environment from the file (as a prerequisite to loading the environment), use the following command:

```
conda env create -f environment.yml
```

Now, to load the environment, use

```
conda activate blank
```

Note that an environment file can be created with the following command:

```
conda env export > environment.yml
```

However, it's often preferable to create an environment file manually, as was done with the file in this project.

Also, these dependencies are also saved in `requirements.txt` for those that would rather use pip. Also, GitHub actions work better with pip, so it's nice to also have the dependencies listed here. This file is created with the following command:

```
pip freeze > requirements.txt
```

### Alternative Quickstart using Conda
Another way to  run code in this repo is to use the following steps.
First, open a terminal and navigate to the root directory of the project and create a conda environment using the following command:
```
conda env create -f environment.yml
```
Now, load the environment with
```
conda activate blank
```
Now, navigate to the directory called `src`
and run
```
doit
```
That should be it!



**Other helpful `conda` commands**

- Create conda environment from file: `conda env create -f environment.yml`
- Activate environment for this project: `conda activate blank`
- Remove conda environment: `conda remove --name blank --all`
- Create blank conda environment: `conda create --name myenv --no-default-packages`
- Create blank conda environment with different version of Python: `conda create --name myenv --no-default-packages python` Note that the addition of "python" will install the most up-to-date version of Python. Without this, it may use the system version of Python, which will likely have some packages installed already.

## `mamba` and `conda` performance issues

Since `conda` has so many performance issues, it's recommended to use `mamba` instead. I recommend installing the `miniforge` distribution. See here: https://github.com/conda-forge/miniforge

