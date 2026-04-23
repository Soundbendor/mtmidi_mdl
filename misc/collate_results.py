import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO
import sys

RESULTS_FOLDER = 'res'
OVERALL_FOLDER = 'res_overall'
CHART_FOLDER = 'res_chart'
PER_MODEL_FOLDER = 'res_model'

datasets = ['polyrhythms', 'dynamics', 'seventh_chords', 'mode_mixture', 'secondary_dominants']

emb_types = ['wav2vec2-large', 'wav2vec2-base', 'MERT-v1-330M', 'MERT-v1-95M', 'jukebox', 'musicgen-large', 'musicgen-medium', 'musicgen-small', 'musicgen-audio', 'baseline-mel', 'baseline-mfcc', 'baseline-chroma', 'baseline-concat']


emb_types_ml = ['wav2vec2-large', 'wav2vec2-base', 'MERT-v1-330M', 'MERT-v1-95M', 'jukebox', 'musicgen-large', 'musicgen-medium', 'musicgen-small']

class_keys = ['loss',
              'layer_idx',
              'accuracy_score',
              'f1_macro',
              'f1_micro',
              'balanced_accuracy_score',
              'aic_pr',
              'aic_pr_avg',
              'bic_pr',
              'bic_pr_avg',
              'ebic_pr',
              'ebic_pr_avg',
              'aic_rep',
              'aic_rep_avg',
              'bic_rep',
              'bic_rep_avg',
              'ebic_rep',
              'ebic_rep_avg',
              'aic_cls',
              'aic_cls_avg',
              'bic_cls',
              'bic_cls_avg',
              'ebic_cls',
              'ebic_cls_avg'
              ]

legend_lr = set(['loss', 'balanced_accuracy_score', 'accuracy_score', 'f1_macro', 'f1_micro'])
emb_idx = {k:i for (i,k) in enumerate(emb_types)}

num_emb = len(emb_types)


class Args:
    def __init__(self, expr_type, dataset, model_size, suffix):
        self.expr_type = expr_type
        self.dataset = dataset
        self.model_size = model_size
        self.suffix = suffix

def df_to_dict(cur_df):
    return {k:v[0] for (k,v) in cur_df.to_dict(as_series=False).items()}

if len(sys.argv) > 1:
    if sys.argv[1] == 'train':
        RESULTS_FOLDER = 'res_train'
        OVERALL_FOLDER = 'res_train_overall'
        CHART_FOLDER = 'res_train_chart'
        PER_MODEL_FOLDER = 'res_train_model'

def get_res_str(model_size, layer_idx, suffix):
    return f'{model_size}_l{layer_idx}-{suffix}.csv'

def get_res_by_layer_idx(expr_type, dataset, model_size, layer_idx, suffix):
    res_dir = UMN.by_projpath_multi(subpaths=[RESULTS_FOLDER, dataset, expr_type],make_dir = False)
    res_str =  get_res_str(model_size, layer_idx, suffix)
    cur_csvf = os.path.join(res_dir, res_str)
    print(f'fetching {cur_csvf}')
    #print(cur_csvf)
    #cur_df = None
    cur_df = pl.read_csv(cur_csvf)
    return df_to_dict(cur_df)

def get_best_results(expr_type, dataset, model_size, suffix):
    args = Args(expr_type, dataset, model_size, suffix)
    cur_study = UO.create_or_load_study(args, seed=UC.SEED, evaluation = True)
    best_param_dict, best_trial_dict, attr_dict = UR.get_best_params_of_study(cur_study)
    cur_params = UR.make_eval_param_dict(best_param_dict, best_trial_dict)
    layer_idx = best_param_dict['layer_idx']['value']
    res = get_res_by_layer_idx(expr_type, dataset, model_size, layer_idx, suffix)
    return res

def collate_best_results():
    suffix = 0
    metrics = {x:{'model': emb_types} for x in class_keys}
    for x in class_keys:
        for ds in datasets:
            if x != 'layer_idx':
                metrics[x][ds] = [-1.0 for _ in range(num_emb)]
            else:
                metrics[x][ds] = [-1 for _ in range(num_emb)]
    for ds in datasets:
        for m in emb_types:
            m_idx = emb_idx[m]
            res = get_best_results('linear', ds, m, suffix)
            for metric,val in res.items():
                # jukebox does not have initial embedding input
                if m == 'jukebox' and 'metric' == 'layer_idx':
                    metrics[metric][ds][m_idx] = val + 1
                else:
                    metrics[metric][ds][m_idx] = val
    
    for metric,mdict in metrics.items():
        cur_df = pl.DataFrame(mdict)
        fname = f'{metric}_overall-{suffix}.csv'
        res_dir = UMN.by_projpath(subpath=OVERALL_FOLDER, make_dir = True)
        cur_csvf = os.path.join(res_dir, fname)
        cur_df.write_csv(cur_csvf)

def plot_per_model_across_ds(cur_x, metric, model_size, suffix, dsdict):
    metric_pprint = UC.CLS_PPRINT[metric]
    model_pprint = UC.MODEL_PPRINT[model_size]
    cur_title = f'{metric_pprint} Across {model_pprint} Layer Indices'
    cur_xlabel = 'Layer Index (1-indexed)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds, dsvals in dsdict.items():
        ds_pprint = UC.DATASET_PPRINT[ds]
        ax.plot(cur_x, dsvals, label=ds_pprint)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(metric_pprint)
    ax.set_title(cur_title)
    cur_loc = 'lower right'
    if metric not in legend_lr:
        cur_loc = 'upper right'
    ax.legend(loc=cur_loc)
    plt.tight_layout()
    res_dir = UMN.by_projpath(subpath=CHART_FOLDER, make_dir = True)
    fname = f'{model_size}-{metric}-{suffix}.png'
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()


def plot_per_ds_across_models(modelx_dict, metric, cur_ds, suffix, modeldict):
    metric_pprint = UC.CLS_PPRINT[metric]
    ds_pprint = UC.DATASET_PPRINT[cur_ds]
    cur_title = f'{metric_pprint} For {ds_pprint} Across Model Depths'
    cur_xlabel = 'Model Depth (percent)'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_size, mvals in modeldict.items():
        model_pprint = UC.MODEL_PPRINT[model_size]
        ax.plot(modelx_dict[model_size], mvals, label=model_pprint)
    ax.set_xlabel(cur_xlabel)
    ax.set_ylabel(metric_pprint)
    ax.set_title(cur_title)
    cur_loc = 'lower right'
    if metric not in legend_lr:
        cur_loc = 'upper right'
    ax.legend(loc=cur_loc)
    plt.tight_layout()
    res_dir = UMN.by_projpath(subpath=CHART_FOLDER, make_dir = True)
    fname = f'{cur_ds}-{metric}-{suffix}.png'
    fpath = os.path.join(res_dir, fname)
    plt.savefig(fpath)
    fig.clear()
    plt.close()


def make_model_compiled_csv_per_ds(model, model_x, metricdict, suffix):
    for ds in datasets:
        df_json = {'layer_number': model_x}
        for x in metricdict.keys():
            df_json[x] = metricdict[x][ds]
        cur_df = pl.DataFrame(df_json)
        fname = f'{model}_compiled-{suffix}.csv'
        res_dir = UMN.by_projpath_multi(subpaths=[PER_MODEL_FOLDER,ds], make_dir = True)
        cur_csvf = os.path.join(res_dir, fname)
        cur_df.write_csv(cur_csvf)



def make_charts(expr_type):
    suffix = 0
    m_dict = {m: {x: {ds: [] for ds in datasets} for x in class_keys if x != 'layer_idx'} for m in emb_types_ml}
    ds_dict = {ds: {x: {m: [] for m in emb_types_ml} for x in class_keys if x != 'layer_idx'} for ds in datasets}
    m_layer_idxs = {} # 0-index layer indices
    m_norm_layer_idxs = {} # scale layer indices to be between 0 and 1
    for m in emb_types_ml:
        start_idx = 0
        num_layers = UC.MODEL_NUM_LAYERS[m] # counting initial embeddings
        cur_lidxs = np.arange(num_layers)
        # non-jukebox models have initial embeddings
        if m != 'jukebox':
            start_idx = 1
            cur_lidxs = np.arange(num_layers-1)
        m_layer_idxs[m] = cur_lidxs + 1
        m_norm_layer_idxs[m] = (cur_lidxs * 100.)/np.max(cur_lidxs)
        for ds in datasets:
            for li in range(start_idx, num_layers):
                res = get_res_by_layer_idx(expr_type, ds, m, li, suffix)
                for metric,val in res.items():
                    if metric != 'layer_idx':
                        m_dict[m][metric][ds].append(val)
                        ds_dict[ds][metric][m].append(val)
    for model,metricdict in m_dict.items():
        cur_x_axis = m_layer_idxs[model] # don't need to normalize layer_idxs
        for metric,dsdict in metricdict.items():
            plot_per_model_across_ds(cur_x_axis, metric, model, suffix, dsdict)
        make_model_compiled_csv_per_ds(model, cur_x_axis, metricdict, suffix)
           
    for cur_ds,metricdict in ds_dict.items():
        for metric, modeldict in metricdict.items():
            plot_per_ds_across_models(m_norm_layer_idxs, metric, cur_ds, suffix, modeldict)



        



collate_best_results()
make_charts('linear')





