
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


def pull_daily_bond_file():
    url = "https://openbondassetpricing.com/wp-content/uploads/2023/12/BondDailyPublicDec2023.csv.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall()
    gzip_file = zip_file.namelist()[0]

    with gzip.open(gzip_file, "rb") as f:
        df = pd.read_csv(f, parse_dates=["trd_exctn_dt"])
    df = df.drop(columns=["Unnamed: 0"])
    # df.info()
    return df


def load_daily_bond(data_dir=DATA_DIR):

    path = data_dir / "manual" / "BondDailyPublic.parquet"
    df = pd.read_parquet(path)
    return df


def _demo():
    df = load_daily_bond(data_dir=DATA_DIR)


if __name__ == "__main__":
    df = pull_daily_bond_file()
    folder_path = DATA_DIR
    folder_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / "manual" / "BondDailyPublic.parquet")
