"""Load project configurations from .env files.
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `config.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.
You can have different sets of variables for difference instances, 
such as `.env.development` or `.env.production`. You would only
need to copy over the settings from one into `.env` to switch
over to the other configuration, for example.

"""
from decouple import config
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = config('DATA_DIR', default=(BASE_DIR / 'data'), cast=Path)
OUTPUT_DIR = config('OUTPUT_DIR', default=(BASE_DIR / 'output'), cast=Path)
LOG_DIR = config('LOG_DIR', default=(BASE_DIR) / 'logs', cast=Path)
WRDS_USERNAME = config("WRDS_USERNAME", default="zhangruoxikathy")
START_DATE = '2003-04-14' # start date in the paper
END_DATE = '2009-06-30'  # end date in the paper

if __name__ == "__main__":
    ## If they don't exist, create the data and output directories
    (DATA_DIR / 'pulled').mkdir(parents=True, exist_ok=True)
    # (DATA_DIR / 'manual').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'pulled' / 'temp').mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'manual').mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
