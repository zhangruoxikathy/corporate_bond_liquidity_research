from pathlib import Path
import pandas as pd
from MakeIntra_Daily_v2_functions import (
    pull_mergent_files, clean_and_filter_mergent_files, pull_TRACE, clean_TRACE)
import config

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)


def pull_intraday_TRACE():
    pull_mergent_files()
    clean_and_filter_mergent_files()
    pull_TRACE()
    clean_TRACE()
    # filter_and_save_intraday_TRACE()


def load_intraday_TRACE(data_dir=DATA_DIR):
    path = data_dir / "pulled" / "intraday_TRACE_filtered.parquet"
    if not path.exists():
        path = data_dir / "manual" / "intraday_TRACE_filtered.parquet"
    df = pd.read_parquet(path)
    return df


def _demo():
    df = load_intraday_TRACE(data_dir=DATA_DIR)


if __name__ == '__main__':
    trade_data = pull_intraday_TRACE()