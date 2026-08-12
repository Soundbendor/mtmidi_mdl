import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO
import sys

thresh_pct = 90
GRAPH_FOLDER = 'res_graph'
suffix = 1
subset = 'train'

models = ["musicgen-small", "musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]

exclude_ds = set(['tempos', 'secondary_dominants', 'mode_mixture'])
root_folder = UC.PROJECT_ROOT

dim_est_names = {
             UC.EFFECTIVE_DIM_N1_NONSTANDARD_FOLDER: "ED (n1, non-std.)",
             UC.BIASED_PART_RTO_NONSTANDARD_FOLDER: "ED (n2, non-std.)",
             UC.PROP_VAR_NONSTANDARD_FOLDER: "Prop. Var. (non-std.)",
             UC.TWONN_FOLDER: "Two-NN (non-std.)"
             }

# skip graphing individual graphs
skip_folders = set([UC.BIASED_PART_RTO_NONSTANDARD_FOLDER, UC.TWONN_FOLDER])

# all layers in one npy file
comp_folders = set([UC.PROP_VAR_NONSTANDARD_FOLDER])

# take last elt
lastelt_folders = set([UC.TWONN_FOLDER])

dim_est_folders = [x for x in dim_est_names.keys()]
res_folder = os.path.join(root_folder, UC.RESULTS_FOLDER)
dses = [x for x in UC.DATASET_SHORT.keys() if x not in exclude_ds]

res = {}
for folder in dim_est_folders:
    res[folder] = {}
    cur_dim_est_folder = os.path.join(root_folder, folder)
    for ds in dses:
        cur_ds_folder = os.path.join(cur_dim_est_folder, ds)
        res[folder][ds] = {}
        for m in models:
            num_layers = UC.MODEL_NUM_LAYERS[m]
            if folder == UC.PROP_VAR_NONSTANDARD_FOLDER:
                cur_fname = f'{m}_thr{thresh_pct}_sd{UC.SEED}_{subset}-{suffix}.npy'
                dim_est_path = os.path.join(cur_ds_folder, cur_fname)
                cur_dim_est = np.load(dim_est_path)
                res[folder][ds][m] = cur_dim_est
            else:
                res[folder][ds][m] = np.zeros(num_layers)
                for l in range(num_layers):
                    cur_fname = f'{m}_l{l}_sd{UC.SEED}_{subset}-{suffix}.npy'
                    dim_est_path = os.path.join(cur_ds_folder, cur_fname)
                    cur_dim_est = np.load(dim_est_path)
                    if folder != UC.TWONN_FOLDER:
                        res[folder][ds][m][l] = cur_dim_est
                    else:
                        res[folder][ds][m][l] = cur_dim_est[-1]

def plot_per_model_across_ds(cur_x, dim_est_type, dim_est_name, model_size, dsdict, prop=False):
    model_pprint = UC.MODEL_PPRINT[model_size]
    ffn_dim = UC.FFN_DIM[model_size]
    cur_title = f'{dim_est_name} Across {model_pprint} Layer Indices'
    subf = dim_est_type
    fname = f'{model_size}-{dim_est_type}-{suffix}.png'
    if prop == True:
        cur_title = f'{dim_est_name} (prop.) Across {model_pprint} Layer Indices'
        subf = f'{dim_est_type}_prop'
        fname = f'{model_size}-{dim_est_type}_prop-{suffix}.png'
    cur_xlabel = 'Layer Index (1-indexed)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, dsvals in dsdict.items():
        ds_pprint = UC.DATASET_PPRINT[ds]
        if prop == False:
            ax.plot(cur_x, dsvals, label=ds_pprint)
        else:
            ax.plot(cur_x, dsvals/ffn_dim, label=ds_pprint)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(dim_est_name)
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
for dim_est_type, dim_est_name in dim_est_names.items():
    if dim_est_type in skip_folders:
        break
    for m in models:
        cur_res = {}
        num_layers = UC.MODEL_NUM_LAYERS[m]
        cur_x = np.arange(num_layers)
        for ds in dses:
            cur_res[ds] = res[dim_est_type][ds][m]
        plot_per_model_across_ds(cur_x, dim_est_type, dim_est_name, m, cur_res, prop = False)
        plot_per_model_across_ds(cur_x, dim_est_type, dim_est_name, m, cur_res, prop = True)

def plot_per_ds_across_models(modelx_dict, dim_est_type, dim_est_name, cur_ds, modeldict, prop = False):
    ds_pprint = UC.DATASET_PPRINT[cur_ds]
    cur_title = f'{dim_est_name} for {ds_pprint} Across Model Depths'
    fname = f'{cur_ds}-{dim_est_type}-{suffix}.png'
    subf = dim_est_type
    if prop == True:
        cur_title = f'{dim_est_name} (prop.) for {ds_pprint} Across Model Depths'
        fname = f'{cur_ds}-{dim_est_type}_prop-{suffix}.png'
        subf = f'{dim_est_type}_prop'
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
    ax.set_ylabel(dim_est_name)
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
for dim_est_type, dim_est_name in dim_est_names.items():
    m_norm_layer_idxs = {}
    for m in models:
        num_layers = UC.MODEL_NUM_LAYERS[m]
        cur_lidxs = np.arange(num_layers)
        m_norm_layer_idxs[m] = (cur_lidxs * 100.)/np.max(cur_lidxs)
    for ds in dses:
        cur_res = {}
        for m in models:
            cur_res[m] = res[dim_est_type][ds][m]
        plot_per_ds_across_models(m_norm_layer_idxs, dim_est_type, dim_est_name, ds, cur_res, prop = False)
        plot_per_ds_across_models(m_norm_layer_idxs, dim_est_type, dim_est_name, ds, cur_res, prop = True)

def plot_per_model_per_ds_across_dim_est(cur_x, cur_ds, model_size, resdict, prop = False):
    model_pprint = UC.MODEL_PPRINT[model_size]
    ds_pprint = UC.DATASET_PPRINT[cur_ds]
    ffn_dim = UC.FFN_DIM[model_size]
    cur_title = f'Est. Dim. For {ds_pprint} Across {model_pprint} Layer Indices'
    fname = f'{model_size}-{cur_ds}-dim_est_all-{suffix}.png'
    subf = 'dim_est-all'
    if prop == True:
        cur_title = f'Est. Dim. For {ds_pprint} (prop.) Across {model_pprint} Layer Indices'
        fname = f'{model_size}-{cur_ds}-dim_est_all_prop-{suffix}.png'
        subf = 'dim_est-all_prop'
    cur_xlabel = 'Layer Index (1-indexed)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for dim_est_type, dim_esttup in resdict.items():
        dim_est_name, prvals = dim_esttup
        if prop == False:
            ax.plot(cur_x, prvals, label=dim_est_name)
        else:
            ax.plot(cur_x, prvals/ffn_dim, label=dim_est_name)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel("Estimated Dimension")
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
        for dim_est_type, dim_est_name in dim_est_names.items():
            cur_res[dim_est_type] = (dim_est_name, res[dim_est_type][ds][m])
        num_layers = UC.MODEL_NUM_LAYERS[m]
        cur_x = np.arange(num_layers)
        plot_per_model_per_ds_across_dim_est(cur_x, ds, m, cur_res, prop = False)
        plot_per_model_per_ds_across_dim_est(cur_x, ds, m, cur_res, prop = True)



    
