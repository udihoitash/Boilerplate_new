from argparse import ArgumentTypeError

import sys
sys.path.append("..")
from utils import connector





def get_table(cfg, limit, offset):
    table = cfg.get('postgres', 'table')
    data = connector.postgres_to_dataframe(table=table)
    return data

