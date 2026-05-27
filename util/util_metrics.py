import csv
import sklearn.metrics as SKM
import matplotlib.pyplot as plt
import numpy as np

from . import util_main as UMN
from . import util_probing as UP
from . import util_constants as UC


# take an array of dicts and turn it in to a dict of arrays
def accum_metrics_to_metric_dict(accum_metrics):
    ret = {}
    for i in range(len(accum_metrics)):
        first_time = False
        if i == 0:
            first_time = True
        for k,v in accum_metrics[i].items():
            # don't save confusion matric
            if k != 'cm':
                if first_time == True:
                    ret[k] = []
                ret[k].append(v)
    return ret

# 0-idxed so preq_idx == num_preq_steps is training on full training set
def is_full_train(preq_idx, datadict):
    return preq_idx == datadict['num_preq_steps']

def get_preq_train_prop(preq_idx, configdict):
    cur_dataset = configdict['dataset']
    cur_prop = UC.DATASET_PREQ_TRAIN_SIZES[cur_dataset][preq_idx]
    return cur_prop

def get_save_other_str(preq_idx, layer_idx):
    return f'p{preq_idx}_l{layer_idx}'

def save_results_to_csv(resdict, configdict, preq_idx, layer_idx):
    res_path = None
    other_str = get_save_other_str(preq_idx, layer_idx)
    res_path = 'res'
    save_path = UMN.get_save_path(res_path, configdict, other=other_str, make_dir = True) 
    cur_header = list(resdict.keys())
    f = open(save_path, 'w')
    csvw = csv.DictWriter(f, fieldnames=cur_header)
    csvw.writeheader()
    csvw.writerow(resdict)
    f.close()

def make_confusion_matrix(truths, preds, preq_idx, layer_idx, datadict, configdict):
    figsize = None
    hide_labels = False
    if datadict['num_classes'] < 10:
        figsize = UC.CM_FIGSIZE_S
    elif datadict['num_classes'] < 30:
        figsize = UC.CM_FIGSIZE_M
    else:
        hide_labels = True
        figsize = UC.CM_FIGSIZE_L
        
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    # convert indices to class strings before feeding into confusion matrix
    cmd = SKM.ConfusionMatrixDisplay.from_predictions(
            [datadict['idxdict'][x] for x in truths],
            [datadict['idxdict'][x] for x in preds],
            labels=datadict['label_arr'],
            normalize='true',
            include_values=hide_labels == False,
            cmap="Purples",
            colorbar=hide_labels == True,
            )

    if hide_labels == True:
        tick_positions = np.arange(0, datadict['num_classes'], 5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=9)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_positions, fontsize=9)

    expr_name = UC.EXPR_PRETTY_NAMES[configdict['expr_type']]
    dataset_name = UC.DATASET_PPRINT[configdict['dataset']]
    cur_train_prop = get_preq_train_prop(preq_idx, configdict)
    cur_train_pct = f'{cur_train_prop * 100:.2f}%'
    title = f'{dataset_name} ({cur_train_pct} of Train Data) {expr_name} Results'
    ax.set_title(title)
    fig.tight_layout()
    other_str = get_save_other_str(preq_idx, layer_idx)
    save_path = UMN.get_save_path('cm', configdict, other=other_str, make_dir = True) 
    plt.savefig(save_path)
    plt.clf()
    return cmd.confusion_matrix


def calculate_log2_nll_sum(nll_arr):
    return np.sum(nll_arr) * UC.LOG2E

# make_cm = make confusion matrix
def get_classification_metrics(truths, preds, nll_arr, preq_idx, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = False, make_cm = False):

    ret = {} 
    ret['log2_nll_sum'] = calculate_log2_nll_sum(nll_arr)
    ret['online_mdl'] = subsetdict['t1logk'] + ret['log2_nll_sum']
    ret['layer_idx'] = layer_idx
    ret['accuracy_score']= SKM.accuracy_score(truths, preds)
    ret['f1_macro'] = SKM.f1_score(truths, preds, average='macro')
    ret['f1_micro'] = SKM.f1_score(truths, preds, average='micro')
    ret['balanced_accuracy_score'] = SKM.balanced_accuracy_score(truths, preds)
    ret['is_full_train'] = is_full_train(preq_idx, datadict)
    ret['train_prop'] = get_preq_train_prop(preq_idx, configdict)

    # only save for eval
    if save_to_csv == True:
        save_results_to_csv(ret, configdict, preq_idx, layer_idx)
    if make_cm == True:
        ret['cm'] = make_confusion_matrix(truths, preds, preq_idx, layer_idx, datadict, configdict)
    return ret

def get_regression_metrics(truths, preds, nll_arr, preq_idx, layer_idx, configdict, save_to_csv = False):
    metrics = ["mean_squared_error",
               "r2_score",
               "mean_absolute_error",
               "explained_variance_score",
               "median_absolute_error",
               "max_error",
               "mean_absolute_percentage_error",
               "root_mean_squared_error",
               "d2_absolute_error_score"
               ]
    ret = {metric: getattr(SKM, name)(truths,preds) for metrics in metrics}
    ret['log2_nll_sum'] = calculate_log2_nll_sum(nll_arr)
    ret['online_mdl'] = subsetdict['t1logk'] + ret['log2_nll_sum']
    ret['layer_idx'] = layer_idx
    ret['is_full_train'] = is_full_train(preq_idx, datadict)
    ret['train_prop'] = get_preq_train_prop(preq_idx, configdict)
    if save_to_csv == True:
        save_results_to_csv(ret, configdict, preq_idx, layer_idx)
    return ret

def get_metrics(truths, preds, nlls, preq_idx, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = False, make_cm = False):
    if datadict['is_classification'] == True:
        return get_classification_metrics(truths, preds, nlls, preq_idx, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = save_to_csv, make_cm = make_cm)
    else:
        return get_regression_metrics(truths, preds, nlls, preq_idx, layer_idx, configdict, save_to_csv = save_to_csv)

def get_optimization_metric(metric_dict, datadict):
    ret = None
    if datadict['is_classification'] == True:
        if datadict['is_balanced'] == True:
            ret = metric_dict['accuracy_score']
        else:
            ret = metric_dict['balanced_accuracy_score']

    else:
        ret = metric_dict['r2_score']
    return ret
