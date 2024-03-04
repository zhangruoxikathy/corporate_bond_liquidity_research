from pathlib import Path
import os
import pandas as pd
from Intraday_TRACE_Pull import pull_TRACE, compile_TRACE
import config

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
FILE_NAME = 'IntradayTRACE.parquet'


def pull_intraday_TRACE():
    pull_TRACE()
    return compile_TRACE()


def load_intraday_TRACE(start_date, end_date, data_dir=DATA_DIR):
    path = data_dir.joinpath(f"pulled/{FILE_NAME}")
    if not path.exists():
        path = data_dir.joinpath(f"manual/{FILE_NAME}")
    df = pd.read_parquet(path)
    df = df[(df['trd_exctn_dt'] >= start_date) & (df['trd_exctn_dt'] <= end_date)]
    return df


def _demo():
    df = load_intraday_TRACE('01-01-2003', '12-31-2009', data_dir=DATA_DIR)


if __name__ == '__main__':
    trade_data = pull_intraday_TRACE()
    trade_data.to_parquet(config.DATA_DIR.joinpath(f'pulled/{FILE_NAME}'))
    os.rmdir(config.DATA_DIR.joinpath(f'pulled/temp'))
