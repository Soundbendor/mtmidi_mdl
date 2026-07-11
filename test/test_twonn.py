from sklearn.datasets import make_swiss_roll
import os
import numpy as np
from two_nn import calc_twonn
import torch
import polars as pl
from skdim.id import TwoNN
from util import util_main as UMN

unused_pct = [0.0, 0.1, 0.25]
test_files = ['swissroll', 'gauss5', 'hc14', 'cauchy20']
#X = cauchy.to_numpy()
#X = gauss.to_numpy()
#X = sr.to_numpy()
n_samples = 64
noise = 0.0
#X,t = make_swiss_roll(n_samples = n_samples, noise = noise)
device = 'cpu'
dupamt = 25
test_path = UMN.by_projpath('test')
if torch.cuda.is_available() == True:
    device = 'cuda'
    torch.cuda.empty_cache()
    torch.set_default_device(device)

for f in test_files:
    for pct in unused_pct:

        dat = pl.read_csv(os.path.join(test_path, f'{f}.txt'), has_header = False, separator=' ')
        X = dat.to_numpy()
        #print(_X.shape)
        #X = np.vstack((dat.to_numpy(), dat.to_numpy()[:dupamt]))
        data = torch.from_numpy(X).to(device)
        slope = calc_twonn(data, batch_size = n_samples, unused_pct = pct, device=device).item()
        print(f'=== {f} ===', f'discard_fraction: {pct}')
        print('mine:', slope)

        twonn = TwoNN(discard_fraction=pct)
        slope2 = twonn.fit_transform(X)
        print('skdim:', slope2)
        print('~~diff~~:', abs(slope - slope2))
