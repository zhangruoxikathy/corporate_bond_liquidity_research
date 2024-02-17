import pandas as pd
import config
import load_wrds_k

OUTPUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR


df = load_wrds_k.load_bondret(data_dir = DATA_DIR)



