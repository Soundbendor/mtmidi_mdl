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
subset = 'train'

models = ["musicgen-small", "musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]

root_folder = UC.PROJECT_ROOT

bpr_folders = [ UC.BIASED_PART_RTO_FOLDER, UC.BIASED_PART_RTO_NONSTANDARD_FOLDER]
bpr_names = {UC.BIASED_PART_RTO_FOLDER: "PR (biased, std.)",
             UC.BIASED_PART_RTO_NONSTANDARD_FOLDER: "PR (biased, non-std.)"
             }

res_folder = os.path.join(root_folder, UC.RESULTS_FOLDER)
dses = [x for x in UC.DATASET_SHORT.keys() if x != 'tempos']

res = {}
for folder in bpr_folders:
    res[folder] = {}
    cur_bpr_folder = os.path.join(root_folder, folder)
    for ds in dses:
        cur_ds_folder = os.path.join(cur_bpr_folder, ds)
        res[folder][ds] = {}
        for m in models:
            num_layers = UC.MODEL_NUM_LAYERS[m]
            res[folder][ds][m] = np.zeros(num_layers)
            for l in range(num_layers):
                cur_fname = f'{m}_l{l}_sd{UC.SEED}_{subset}-{suffix}.npy'
                bpr_path = os.path.join(cur_ds_folder, cur_fname)
                cur_bpr = np.load(bpr_path)
                res[folder][ds][m][l] = cur_bpr

def plot_per_model_across_ds(cur_x, bpr_type, bpr_name, model_size, dsdict, prop=False):
    model_pprint = UC.MODEL_PPRINT[model_size]
    ffn_dim = UC.FFN_DIM[model_size]
    cur_title = f'{bpr_name} Across {model_pprint} Layer Indices'
    subf = bpr_type
    fname = f'{model_size}-{bpr_type}-{suffix}.png'
    if prop == True:
        cur_title = f'{bpr_name} (prop.) Across {model_pprint} Layer Indices'
        subf = f'{bpr_type}_prop'
        fname = f'{model_size}-{bpr_type}_prop-{suffix}.png'
    cur_xlabel = 'Layer Index (1-indexed)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, dsvals in dsdict.items():
        ds_pprint = UC.DATASET_PPRINT[ds]
        if prop == False:
            ax.plot(cur_x, dsvals, label=ds_pprint)
        else:
            ax.plot(cur_x, dsvals/ffn_dim, label=ds_pprint)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(bpr_name)
    ax.set_title(cur_title)
    cur_loc = 'center right'
    ax.legend(loc=cur_loc, bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    
    res_dir = UMN.by_projpath_multi(subpaths=[GRAPH_FOLDER, subf], make_dir = True)
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()


# cross-dataset per model per pr type
for bpr_type, bpr_name in bpr_names.items():
    for m in models:
        cur_res = {}
        num_layers = UC.MODEL_NUM_LAYERS[m]
        cur_x = np.arange(num_layers)
        for ds in dses:
            cur_res[ds] = res[bpr_type][ds][m]
        plot_per_model_across_ds(cur_x, bpr_type, bpr_name, m, cur_res, prop = False)
        plot_per_model_across_ds(cur_x, bpr_type, bpr_name, m, cur_res, prop = True)

def plot_per_ds_across_models(modelx_dict, bpr_type, bpr_name, cur_ds, modeldict, prop = False):
    ds_pprint = UC.DATASET_PPRINT[cur_ds]
    cur_title = f'{bpr_name} for {ds_pprint} Across Model Depths'
    fname = f'{cur_ds}-{bpr_type}-{suffix}.png'
    subf = bpr_type
    if prop == True:
        cur_title = f'{bpr_name} (prop.) for {ds_pprint} Across Model Depths'
        fname = f'{cur_ds}-{bpr_type}_prop-{suffix}.png'
        subf = f'{bpr_type}_prop'
    cur_xlabel = 'Model Depth (percent)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_size, mvals in modeldict.items():
        model_pprint = UC.MODEL_PPRINT[model_size]
        ffn_dim = UC.FFN_DIM[model_size]
        if prop == False:
            ax.plot(modelx_dict[model_size], mvals, label=model_pprint)
        else:
            ax.plot(modelx_dict[model_size], mvals/ffn_dim, label=model_pprint)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(bpr_name)
    ax.set_title(cur_title)
    cur_loc = 'center right'
    ax.legend(loc=cur_loc, bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    res_dir = UMN.by_projpath_multi(subpaths=[GRAPH_FOLDER, subf], make_dir = True)
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()

# cross-model per dataset per type
for bpr_type, bpr_name in bpr_names.items():
    m_norm_layer_idxs = {}
    for m in models:
        num_layers = UC.MODEL_NUM_LAYERS[m]
        cur_lidxs = np.arange(num_layers)
        m_norm_layer_idxs[m] = (cur_lidxs * 100.)/np.max(cur_lidxs)
    for ds in dses:
        cur_res = {}
        for m in models:
            cur_res[m] = res[bpr_type][ds][m]
        plot_per_ds_across_models(m_norm_layer_idxs, bpr_type, bpr_name, ds, cur_res, prop = False)
        plot_per_ds_across_models(m_norm_layer_idxs, bpr_type, bpr_name, ds, cur_res, prop = True)

def plot_per_model_per_ds_across_bpr(cur_x, cur_ds, model_size, resdict, prop = False):
    model_pprint = UC.MODEL_PPRINT[model_size]
    ds_pprint = UC.DATASET_PPRINT[cur_ds]
    ffn_dim = UC.FFN_DIM[model_size]
    cur_title = f'PRs For {ds_pprint} Across {model_pprint} Layer Indices'
    fname = f'{model_size}-{cur_ds}-bpr_all-{suffix}.png'
    subf = 'b_pr-all'
    if prop == True:
        cur_title = f'PRs For {ds_pprint} (prop.) Across {model_pprint} Layer Indices'
        fname = f'{model_size}-{cur_ds}-bpr_all_prop-{suffix}.png'
        subf = 'b_pr-all_prop'
    cur_xlabel = 'Layer Index (1-indexed)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for bpr_type, bprtup in resdict.items():
        bpr_name, prvals = bprtup
        if prop == False:
            ax.plot(cur_x, prvals, label=bpr_name)
        else:
            ax.plot(cur_x, prvals/ffn_dim, label=bpr_name)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(bpr_name)
    ax.set_title(cur_title)
    cur_loc = 'center right'
    ax.legend(loc=cur_loc, bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    res_dir = UMN.by_projpath_multi(subpaths=[GRAPH_FOLDER, subf], make_dir = True)
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()

# cross-prtype per model per ds
for m in models:
    for ds in dses:
        cur_res = {}
        for bpr_type, bpr_name in bpr_names.items():
            cur_res[bpr_type] = (bpr_name, res[bpr_type][ds][m])
        num_layers = UC.MODEL_NUM_LAYERS[m]
        cur_x = np.arange(num_layers)
        plot_per_model_per_ds_across_bpr(cur_x, ds, m, cur_res, prop = False)
        plot_per_model_per_ds_across_bpr(cur_x, ds, m, cur_res, prop = True)



    
