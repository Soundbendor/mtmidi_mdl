import csv
import sklearn.metrics as SKM
import matplotlib.pyplot as plt
import numpy as np

from . import util_main as UMN
from . import util_probing as UP
from . import util_constants as UC


def aic(nll, k, N, per_sample = False):
    res = 2. * nll + 2.*k
    if per_sample == True:
        res /= N
    return res

def bic(nll, k, N, per_sample = False):
    res = 2. * nll + k * np.log(N)
    if per_sample == True:
        res /= N
    return res

def ebic(nll, k, N, D, gamma = UC.EBIC_GAMMA, per_sample = False):
    res = 2. * nll + k * np.log(N) + 2 * gamma * k* np.log(D)
    if per_sample == True:
        res /= N
    return res

def get_save_other_str(layer_idx):
    return f'l{layer_idx}'

def save_results_to_csv(resdict, configdict, layer_idx):
    res_path = None
    other_str = 'best'
    if configdict['eval_best'] == False:
        other_str = get_save_other_str(layer_idx)
    if configdict['eval_nll'] == True:
        res_path = 'res_train'
    else:
        res_path = 'res'
    save_path = UMN.get_save_path(res_path, configdict, other=other_str, make_dir = True) 
    cur_header = list(resdict.keys())
    f = open(save_path, 'w')
    csvw = csv.DictWriter(f, fieldnames=cur_header)
    csvw.writeheader()
    csvw.writerow(resdict)
    f.close()

def make_confusion_matrix(truths, preds, layer_idx, datadict, configdict):
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
    title = f'{dataset_name} {expr_name} Results'
    ax.set_title(title)
    fig.tight_layout()
    other_str = 'best'
    if configdict['eval_best'] == False:
        other_str = get_save_other_str(layer_idx)
    save_path = UMN.get_save_path('cm', configdict, other=other_str, make_dir = True) 
    plt.savefig(save_path)
    plt.clf()
    return cmd.confusion_matrix

# make_cm = make confusion matrix
def get_classification_metrics(truths, preds, loss, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = False, make_cm = False):

    ret = {} 
    ret['loss'] = loss
    ret['layer_idx'] = layer_idx
    ret['accuracy_score']= SKM.accuracy_score(truths, preds)
    ret['f1_macro'] = SKM.f1_score(truths, preds, average='macro')
    ret['f1_micro'] = SKM.f1_score(truths, preds, average='micro')
    ret['balanced_accuracy_score'] = SKM.balanced_accuracy_score(truths, preds)
    # only save for eval
    if save_to_csv == True:
        N = None
        pr = UP.load_part_rto(configdict, trial_number, layer_idx)
        if configdict['eval_nll'] == True:
            N = subsetdict['train_size']
        else:
            N = subsetdict['test_size']
        model_size = configdict['model_size']
        D = UC.FFN_DIM[model_size] 
        c = datadict['num_classes']
        k = (D*c) + c
        ret['aic_rep'] = aic(loss, D, N, per_sample = False)
        ret['aic_rep_avg'] = aic(loss, D, N, per_sample = True)
        ret['bic_rep'] = bic(loss, D, N, per_sample = False) 
        ret['bic_rep_avg'] = bic(loss, D, N, per_sample = True) 
        ret['ebic_rep'] = ebic(loss, D, N, D, gamma = UC.EBIC_GAMMA, per_sample = False) 
        ret['ebic_rep_avg'] = ebic(loss, D, N, D, gamma = UC.EBIC_GAMMA, per_sample = True) 

        ret['aic_cls'] = aic(loss, k, N, per_sample = False)
        ret['aic_cls_avg'] = aic(loss, k, N, per_sample = True)
        ret['bic_cls'] = bic(loss, k, N, per_sample = False) 
        ret['bic_cls_avg'] = bic(loss, k, N, per_sample = True) 
        ret['ebic_cls'] = ebic(loss, k, N, k, gamma = UC.EBIC_GAMMA, per_sample = False) 
        ret['ebic_cls_avg'] = ebic(loss, k, N, k, gamma = UC.EBIC_GAMMA, per_sample = True) 
        
        ret['aic_pr'] = aic(loss, pr, N, per_sample = False)
        ret['aic_pr_avg'] = aic(loss, pr, N, per_sample = True)
        ret['bic_pr'] = bic(loss, pr, N, per_sample = False) 
        ret['bic_pr_avg'] = bic(loss, pr, N, per_sample = True) 
        ret['ebic_pr'] = ebic(loss, pr, N, pr, gamma = UC.EBIC_GAMMA, per_sample = False) 
        ret['ebic_pr_avg'] = ebic(loss, pr, N, pr, gamma = UC.EBIC_GAMMA, per_sample = True) 


        save_results_to_csv(ret, configdict, layer_idx)
    if make_cm == True:
        ret['cm'] = make_confusion_matrix(truths, preds, layer_idx, datadict, configdict)
    return ret

def get_regression_metrics(truths, preds, loss, layer_idx, configdict, save_to_csv = False):
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
    ret['loss'] = loss
    ret['layer_idx'] = layer_idx
    if save_to_csv == True:
        save_results_to_csv(ret, configdict, layer_idx)
    return ret

def get_metrics(truths, preds, loss, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = False, make_cm = False):
    if datadict['is_classification'] == True:
        return get_classification_metrics(truths, preds, loss, layer_idx, trial_number, datadict, subsetdict, configdict, save_to_csv = save_to_csv, make_cm = make_cm)
    else:
        return get_regression_metrics(truths, preds, loss, layer_idx, configdict, save_to_csv = save_to_csv)

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
