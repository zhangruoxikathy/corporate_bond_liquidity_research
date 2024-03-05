
'''
Overview
-------------
This Python script aims to cleaned daily TRACE data from https://openbondassetpricing.com/
 
Requirements
-------------
Access to https://openbondassetpricing.com/

Package versions 
-------------
pandas v1.4.4
numpy v1.21.5
wrds v3.1.2
datetime v3.9.13
csv v1.0
gzip v3.9.13
'''

#* ************************************** */
#* Libraries                              */
#* ************************************** */ 
import pandas as pd
import numpy as np

import requests
import zipfile
import io
import gzip

import config
from pathlib import Path

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
BOND_DAILY_FILENAME = "BondDailyPublic.parquet"
WRDS_MMN_FILENAME = "WRDS_MMN_Corrected_Data.parquet"


def pull_daily_bond_file():
    url = "https://openbondassetpricing.com/wp-content/uploads/2023/12/BondDailyPublicDec2023.csv.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall(DATA_DIR.joinpath("pulled/temp"))
    gzip_file = DATA_DIR.joinpath(f"pulled/temp/{zip_file.namelist()[0]}")

    df = pd.read_csv(gzip_file, parse_dates=["trd_exctn_dt"], compression='gzip')
    df = df.drop(columns=["Unnamed: 0"])
    return df


def pull_mmn_corrected_bond_file():
    url = "https://openbondassetpricing.com/wp-content/uploads/2023/10/WRDS_MMN_Corrected_Data.csv.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall(DATA_DIR.joinpath("pulled/temp"))
    gzip_file = DATA_DIR.joinpath(f"pulled/temp/{zip_file.namelist()[0]}")

    df = pd.read_csv(gzip_file, parse_dates=["date"], compression='gzip')
    df = df.drop(columns=["Unnamed: 0"])
    return df


def load_daily_bond(data_dir=DATA_DIR):
    path = data_dir / "pulled" / BOND_DAILY_FILENAME
    if not path.exists():
        path = data_dir / "manual" / BOND_DAILY_FILENAME
    df = pd.read_parquet(path)
    return df


def load_mmn_corrected_bond(data_dir=DATA_DIR):
    path = data_dir / "pulled" / WRDS_MMN_FILENAME
    if not path.exists():
        path = data_dir / "manual" / WRDS_MMN_FILENAME
    df = pd.read_parquet(path)
    return df


def _demo():
    df = load_daily_bond(data_dir=DATA_DIR)
    df_mmn = load_mmn_corrected_bond(data_dir=DATA_DIR)


if __name__ == "__main__":
    folder_path = DATA_DIR
    folder_path.mkdir(parents=True, exist_ok=True)
    df = pull_daily_bond_file()
    df.to_parquet(DATA_DIR / "pulled" / "BondDailyPublic.parquet")
    del df

    df_mmn = pull_mmn_corrected_bond_file()
    df_mmn.to_parquet(DATA_DIR / "pulled" / "WRDS_MMN_Corrected_Data.parquet")
