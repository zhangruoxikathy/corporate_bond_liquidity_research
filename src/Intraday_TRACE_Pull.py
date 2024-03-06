"""
A refactorization to add parallel execution to the source code
by Alexander Dickerson.
https://github.com/Alexander-M-Dickerson/TRACE-corporate-bond-processing/tree/main
"""



import logging
from functools import partial
import wrds
import glob

import config
import json
import load_opensource
import pandas as pd

import gc

log_format = '%(asctime)s - %(name)s:%(levelname)s - %(process)d - %(message)s'
log_filename = config.LOG_DIR / 'PullIntraday_WRDS.log'
logging.basicConfig(level=logging.DEBUG,
                    format=log_format,
                    datefmt='%y-%m-%d %H:%M',
                    filename=log_filename,
                    encoding='utf-8',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def deprecated(msg):
    def outer(func):
        def inner(*args, **kwargs):
            print(f"{func.__name__} has been deprecated: {msg}")
    return outer


@deprecated('This function is no longer needed to get CUSIPs. Use _get_filter_parameters instead')
def pull_mergent_files(DIR):
    """ Download Mergent Files from WRDS.
        This function is needed to get the files required to determine the target cusips
    """
    with wrds.Connection(wrds_username=config.WRDS_USERNAME) as db:

        fisd_issuer = db.raw_sql("""SELECT issuer_id,country_domicile
                          FROM fisd.fisd_mergedissuer
                          """)

        logging.info(f"Pulled and saved fisd_issuer file from wrds.")

        fisd_issue = db.raw_sql("""SELECT complete_cusip, issue_id,
                          issuer_id, foreign_currency,
                          coupon_type,coupon,convertible,
                          asset_backed,rule_144a,
                          bond_type,private_placement,
                          interest_frequency,dated_date,
                          day_count_basis,offering_date
                          FROM fisd.fisd_mergedissue
                          """)
        logging.info(f"Pulled and saved fisd_issue file from wrds")

    fisd_issuer.to_csv(DIR.joinpath("fisd_issuer_file.csv"), index=False)
    fisd_issue.to_csv(DIR.joinpath("fisd_issue_file.csv"), index=False)


def _pull_trace_data_from_wrds(db, chunk, start_date, end_date):
    parm = {
        'cusip_id': tuple(chunk),
        'start_date': start_date,
        'end_date': end_date
    }

    # original query:
    # f"SELECT cusip_id, bond_sym_id, trd_exctn_dt, trd_exctn_tm, days_to_sttl_ct, lckd_in_ind, wis_fl,"
    # f"sale_cndtn_cd, msg_seq_nb, trc_st, trd_rpt_dt, trd_rpt_tm, entrd_vol_qt, rptd_pr, yld_pt, asof_cd,"
    # f"orig_msg_seq_nb, rpt_side_cd, cntra_mp_id "
    # f"FROM trace.trace_enhanced "
    # f"WHERE cusip_id in %(cusip_id)s AND trd_exctn_dt >= %(start_date)s AND trd_exctn_dt <= %(end_date)s",
    try:
        trace = db.raw_sql(
            f"SELECT cusip_id, trd_exctn_dt, trd_exctn_tm, days_to_sttl_ct, lckd_in_ind, wis_fl,"
            f"msg_seq_nb, entrd_vol_qt, rptd_pr, orig_msg_seq_nb "
            f"FROM trace.trace_enhanced "
            f"WHERE cusip_id in %(cusip_id)s AND trd_exctn_dt >= %(start_date)s AND trd_exctn_dt <= %(end_date)s",
            params=parm)
        return trace
    except Exception as e:
        logging.exception(f"Error pulling chunk {chunk[0]}")


def _save_trace_data(chunk_id, df):
    save_path = config.DATA_DIR.joinpath(f"pulled/temp/intraday_{chunk_id}.parquet")
    if not df.empty:
        df.to_parquet(save_path)
        return True
    return False


def _pull_and_save_trace_data(chunk, db, start_date, end_date):
    df = _pull_trace_data_from_wrds(db, chunk[1], start_date, end_date)
    if not _save_trace_data(chunk[0], df):
        logging.info(f"Chunk {chunk[0]} contained no data")
    logging.info(f"Pulled and saved cusip chunk {chunk[0]} from wrds")


def _pull_TRACE_sequential(db, cusip_chunks, start_date, end_date):
    """ Pulls and saves TRACE data sequentially """
    for chunk in cusip_chunks:
        partial_query_func = partial(_pull_and_save_trace_data, db=db, start_date=start_date, end_date=end_date)
        partial_query_func(chunk)


def _pull_TRACE_parallel(db, cusip_chunks, start_date, end_date):
    """ Pulls and saves TRACE data in parallel """
    # Initialize a Pool of workers
    with Pool(config.NUM_WORKERS) as pool:
        # Use partial to pass additional arguments to _wrds_query_and_save_data
        partial_query_func = partial(_pull_and_save_trace_data, db=db, start_date=start_date, end_date=end_date)
        # Map the function to the chunks of CUSIPs
        pool.map(partial_query_func, cusip_chunks)


def _chunk_cusips(cusips, n=100):
    """ Divide cusips into chunks of size n, returning a list of tuple of chunk id and chunk """
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield i // n, l[i:i + n]

    return list(divide_chunks(cusips, n))


def pull_TRACE():
    """ Pulls and saves TRACE data from WRDS into chunks"""
    cusips = _get_cusips_from_monthly()
    start_date = '01-01-2003'
    end_date = '12-31-2023'
    cusip_chunks = _chunk_cusips(cusips)
    with wrds.Connection(wrds_username=config.WRDS_USERNAME) as db:
        if config.RUN_TRACE_IN_PARALLEL:
            _pull_TRACE_parallel(db=db,
                                 cusip_chunks=cusip_chunks,
                                 start_date=start_date,
                                 end_date=end_date)
        else:
            _pull_TRACE_sequential(db=db,
                                   cusip_chunks=cusip_chunks,
                                   start_date=start_date,
                                   end_date=end_date)


def _get_cusips_from_monthly():
    filters_path = config.DATA_DIR.joinpath('pulled/filters.json')
    if filters_path.exists():
        contents = ""
        with open(filters_path, 'r') as f:
            contents = json.load(f)
        if contents:
            return contents['cusips']

    df_daily = load_opensource.load_daily_bond(data_dir=config.DATA_DIR)
    df_bondret = load_wrds_bondret.load_bondret(data_dir=config.DATA_DIR)
    merged_df = data.all_trace_data_merge(df_daily, df_bondret)

    del df_daily
    del df_bondret
    gc.collect()

    merged_df = data.sample_selection(merged_df)

    return merged_df['cusip'].unique()

def compile_TRACE():
    path = config.DATA_DIR.joinpath('pulled/temp/intraday_[0-9]*.parquet')
    files = glob.glob(str(path))
    dfs = []
    for file in files:
        df = pd.read_parquet(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df
