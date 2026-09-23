import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO


dses = ["chords","mode_mixture","scales","simple_progressions","dynamics","notes","secondary_dominants","time_signatures","intervals","polyrhythms","seventh_chords"]

emb_types = ['wav2vec2-large', 'wav2vec2-base', 'MERT-v1-330M', 'MERT-v1-95M', 'musicgen-large', 'musicgen-medium', 'musicgen-small']

seed = 39
suffix = 1

stats = ['max', 'min', 'mean', 'std']

CHART_FOLDER = 'emb_stats'
def plot_over_dses_per_model(_cur, stat, model_size, num_layers, dim):
    model_pprint = UC.MODEL_PPRINT[model_size]
    pretty_stat = stat.capitalize()

    accum_type = 'Max'
    if stat == 'mean':
        accum_type = 'Avg.'
    cur_title = f'{pretty_stat} for {model_pprint} {accum_type} Across Datasets Per Layer'
    cur_xlabel = 'Vector Index'
    fig, ax = plt.subplots(figsize=(7, 5))
    for l in range(num_layers):
        ax.plot(np.arange(dim), _cur[l])
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(pretty_stat)
    ax.set_title(cur_title)
    #cur_loc = 'lower right'
    """
    if metric not in legend_lr:
        cur_loc = 'upper right'
    ax.legend(loc=cur_loc)
    """
    plt.tight_layout()
    res_dir = UMN.by_projpath(subpath=CHART_FOLDER, make_dir = True)
    fname = f'{model_size}-{stat}-{suffix}.png'
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()



            


for stat in stats:
    for m in emb_types:
        num_layers = UC.MODEL_NUM_LAYERS[m] # counting initial embeddings
        ffn_dim = UC.FFN_DIM[m] 
        cur = np.ones((num_layers, ffn_dim))
        if stat == 'min':
            cur = cur * np.inf
        elif stat == 'max':
            cur = cur * (-np.inf)
        else:
            cur = cur * 0.
        for l in range(num_layers):
            for di, ds in enumerate(dses):
                res_dir = UMN.by_projpath(subpath='data_stats', make_dir = False)
                cur_d = os.path.join(res_dir, ds)
                cur_fname = f"{m}_l{l}_sd{seed}_train-{stat}-{suffix}.npy"
                cur_fp = os.path.join(cur_d, cur_fname)
                x = np.load(cur_fp)
                if di == 0 and l == 0:
                    cur[0] = x
                else:
                    if stat == 'min':
                        cur[l] = np.minimum(cur[l], x)
                    elif stat == 'max' or stat == 'std':
                        cur[l] = np.maximum(cur[l], x)
                    elif stat == 'mean':
                        cur[l] = np.mean(np.vstack((cur[l], x)), axis=0)

        plot_over_dses_per_model(cur, stat, m, num_layers, ffn_dim)




