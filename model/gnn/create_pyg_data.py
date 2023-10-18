import os
import os.path as osp
import shutil
from typing import Callable, List, Optional
import pandas as pd
import torch
import json
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor
)
import argparse
from s2orc_local import S2ORC



parser = argparse.ArgumentParser(
                    prog='EXPERT HGT',
                    description='HGT Model Pretraining with EXPERT MLKG',
                    epilog='Text at the bottom of help')
parser.add_argument('--raw_data_dir',help="Location of the Raw EXPERT context graphs")
args = parser.parse_args()
dataset = S2ORC(args.raw_data_dir)