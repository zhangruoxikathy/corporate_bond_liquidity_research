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


def task_setup():
    pass