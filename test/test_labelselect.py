import torch, torch.utils.data as TUD
from torch import nn
import optuna, pickle, numpy as np  

import util.util_main as UMN
import util.util_metrics as UME
import util.util_constants as UC
import util.util_data as UD
import util.util_wandb as UW
import util.util_optuna as UO
import util.util_probing as UP
import util.util_rdb as UR

from probe_dataset import ProbeDataset

from functools import partial
from distutils.util import strtobool
import os, sys, time, argparse, copy

device = 'cpu'
if torch.cuda.is_available() == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)


DS = 'time_signatures'
FROM_DIR = f'/home/dxk/osu/mtmidi_mdl/'

datadict = UD.load_data_dict(DS)
cur_ds = ProbeDataset(datadict, 'MERT-v1-95M', layer_idx=10, from_dir = FROM_DIR, to_torch = True, device = device)


torch_gen = torch.Generator(device=device)
torch_gen.manual_seed(39)

#for i in range(datadict['num_classes']):
i = 0
train_dl = TUD.DataLoader(cur_ds, batch_size = datadict['num_examples'], shuffle=False, generator=torch_gen)

with torch.no_grad():
    for batch_idx, data in enumerate(train_dl):
        _ipt, ground_truth = data
        #print(ground_truth, ground_truth.shape, ground_truth.min(), ground_truth.max(), datadict['num_classes'])
        cur_idxs = torch.where(ground_truth == i)[0]
        ipt = _ipt[cur_idxs]
        print(cur_idxs.shape, _ipt.shape, ipt.shape)

