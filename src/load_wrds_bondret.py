
'''
Overview
-------------
This Python script aims to load corporate bond data from WRDSAPPS.BONDRET 
on WRDS since 2002-07.
 
Requirements
-------------
Access to the WRDS server and associated databases.

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
import wrds
import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd
import numpy as np
import config
from pathlib import Path


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME
# START_DATE = config.START_DATE
# END_DATE = config.END_DATE



# https://wrds-www.wharton.upenn.edu/pages/get-data/wrds-bond-returns/wrds-bond-returns/
description_bondret = {
    "cusip": "CUSIP ID",
    "date": "date",
    "issue_id": "Mergent FISD Issue Id",
    "bond_sym_id": "TRACE Bond Symbol",
    "bsym": "Bloomberg Identifier",
    "isin": "ISIN",
    "company_symbol": "Company Symbol (issuer stock ticker)",
    "bond_type": "Corporate Bond Types: Convertible, Debenture, Medium Term Note, MTN Zero",
    "security_level": "Indicates if the security is a secured, senior or subordinated issue of the issuer",
    "conv": "Flag Convertible",
    "offering_date": "offering_date",
    "offering_amt": "offering_amt",
    "offering_price": "offering_price",
    "principal_amt": "principal_amt",
    "maturity": "maturity",
    "treasury_maturity": "treasury_maturity",
    "coupon": "coupon",
    "day_count_basis": "day_count_basis",
    "dated_date": "dated_date",
    "first_interest_date": "first_interest_date",
    "last_interest_date": "last_interest_date",
    "ncoups": "ncoups",
    "amount_outstanding": "amount_outstanding",
    "n_mr": "numeric moody rating",
    "tmt": 'time to maturity'
}


def pull_bondret(wrds_username=WRDS_USERNAME):
    """
    Pull corporate bond returns data from WRDS.
    See description_bondret for a description of the variables.
    """

    # original query fields
    # cusip, date, price_eom, price_ldm, price_l5m,
    # bsym, isin, company_symbol, bond_type, rating_cat, tmt,
    # rating_class, t_date, t_volume, t_dvolume, t_spread,
    # security_level, conv, offering_date, offering_amt, offering_price,
    # principal_amt, maturity, treasury_maturity, coupon, day_count_basis,
    # dated_date, first_interest_date, last_interest_date, ncoups,
    # amount_outstanding, r_sp, r_mr, r_fr, n_sp, n_mr, n_fr, rating_num

    sql_query = """
        SELECT 
            cusip, date, price_eom, tmt,
            t_volume, t_dvolume, t_spread,
            offering_amt, offering_price,
            principal_amt, maturity, coupon, ncoups,
            amount_outstanding, r_mr, n_mr, offering_date
        FROM 
            WRDSAPPS.BONDRET
        WHERE 
            date >= '07/01/2002'
        """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     comp = db.raw_sql(sql_query, date_cols=["date"])
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    bond = db.raw_sql(sql_query, date_cols=["date"])
    db.close()

    bond["year"] = bond["date"].dt.year
    return bond


def load_bondret(data_dir=DATA_DIR):
    path = Path(data_dir) / "pulled" / "Bondret.parquet"
    bond = pd.read_parquet(path)
    return bond


def _demo():
    comp = load_bondret(data_dir=DATA_DIR)


if __name__ == "__main__":
    comp = pull_bondret(wrds_username=WRDS_USERNAME)
    comp.to_parquet(DATA_DIR / "pulled" / "Bondret.parquet")
