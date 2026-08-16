import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO
import sys


GRAPH_FOLDER = 'res_graph'
suffix = 1
num_coeffs = 3
subset = 'train'
GRAPH_SUBFOLDER = 'pca'
models = ["musicgen-small", "musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]
#models = ["MERT-v1-95M"]
#models = ["musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]

root_folder = UC.PROJECT_ROOT

colors = ["#00ffff", "#04d8b2", "#069af3", "#e6daa6", "#000000", "#0343df", "#a52a2a", "#7fff00", "#ff7f50", "#006400", "#ff00ff", "#ffd700", "#380282", "#f0e68c", "#c79fef", "#7bc8f6", "#c20078", "#808000", "#f97306", "#da70d6", "#dda0dd", "#800080", "#ff0000", "#fa8072", "#c0c0c0", "#d1b26f", "#029386", "#40e0d0", "#fbdd7e", "#ffff00", "#9acd32", "#c20078", "#ff81c0", "#f5deb3"]
exclude_ds = set(['tempos'])
res_folder = os.path.join(root_folder, UC.RESULTS_FOLDER)
#dses = [x for x in UC.DATASET_SHORT.keys() if x not in exclude_ds]
dses = ["polyrhythms", "dynamics", "notes", "scales", "seventh_chords", "time_signatures", "intervals", "simple_progressions", "chords"]

coeff_folder = os.path.join(root_folder, UC.PCA_COEFFS_NONSTANDARD_FOLDER)
clidx_folder = os.path.join(root_folder, UC.PCA_CLASS_IDXS_NONSTANDARD_FOLDER)


def plot_pca_coeffs(ds, model_size, layer_idx, coeffs, clidxs, max_idx):
    model_pprint = UC.MODEL_PPRINT[model_size]
    ffn_dim = UC.FFN_DIM[model_size]
    ds_pprint = UC.DATASET_PPRINT[ds]
    cur_title = f'{ds_pprint} ({model_pprint}, Layer {layer_idx}) Proj. Onto PCs'
    fname = f'l{layer_idx}_sd{UC.SEED}_{subset}-{suffix}.png'
    ylab = 'ID'
    cur_xlabel = 'Layer Index (1-indexed)'
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(max_idx+1):
        want_idxs = np.where(clidxs == i)
        want_coeffs = coeffs[want_idxs]
        ax.scatter(want_coeffs[:,2], want_coeffs[:,1], want_coeffs[:,0], c=colors[i])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(cur_title)
    #cur_loc = 'center right'
    #ax.legend(loc=cur_loc, bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    
    res_dir = UMN.by_projpath_multi(subpaths=[GRAPH_FOLDER, GRAPH_SUBFOLDER, ds, model_size], make_dir = True)
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()



for ds in dses:
    coeff_ds_folder = os.path.join(coeff_folder, ds)
    clidx_ds_folder = os.path.join(clidx_folder, ds)
    for m in models:
        num_layers = UC.MODEL_NUM_LAYERS[m]
        for l in range(num_layers):
        #for l in [10]:
            cur_fname = f'{m}_l{l}_nc{num_coeffs}_sd{UC.SEED}_{subset}-{suffix}.npy'
            coeff_path = os.path.join(coeff_ds_folder, cur_fname)
            clidx_path = os.path.join(clidx_ds_folder, cur_fname)
            cur_coeffs = np.load(coeff_path)
            cur_clidxs = np.load(clidx_path)
            max_idx = cur_clidxs.max()
            plot_pca_coeffs(ds, m, l, cur_coeffs, cur_clidxs, max_idx)
            

