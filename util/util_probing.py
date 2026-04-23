import copy
import os

from . import util_main as UMN
from . import util_constants as UC

import numpy as np
import polars as pl
import torch, torch.utils.data as TUD
from sklearn.model_selection import train_test_split

def get_train_test_splits(datadict, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = UC.SEED):
    ret = {}
    if datadict['train_on_middle'] == True:
        num_examples = datadict['num_examples']
        rng = np.random.default_rng(seed)
        _label = datadict['label']
        _idxs = np.arange(num_examples)
        temp_df = pl.DataFrame({'label': datadict['df'][_label], 'idxs': _idxs})
        temp_df = temp_df.sort(_label)
        idxs = temp_df['idxs'].to_numpy()
        test_pct = 1. - train_pct
        first_prop = test_pct * 0.5
        last_prop = 1. - first_prop
        up_to_middle = int(first_prop * num_examples)
        middle_to_end = int(last_prop * num_examples)
        ret['train_idxs'] = idxs[up_to_middle:middle_to_end]
        if test_subpct < 1.:
            ends = np.hstack([idxs[:up_to_middle], idxs[middle_to_end:]])
            validtest = train_test_split(ends, train_size = test_subpct, random_state = seed, shuffle = True)
            ret['test_idxs'] = validtest[0]
            ret['valid_idxs'] = validtest[1]
        else:
            ret['test_idxs'] = np.hstack([idxs[:up_to_middle], idxs[middle_to_end:]])
            ret['valid_idxs'] = np.array([])
    else:
        idxs = np.arange(datadict['num_examples'])
        labels = datadict['df'][datadict['label']].to_numpy()
        cur_train_validtest = train_test_split(idxs, train_size = train_pct, random_state = seed, shuffle = True, stratify = labels)
        ret['train_idxs'] = cur_train_validtest[0]
        if test_subpct < 1.:
            validtest_idxs = cur_train_validtest[1]
            validtest_labels = labels[validtest_idxs]
            cur_validtest = train_test_split(validtest_idxs, train_size = test_subpct, random_state = seed, shuffle = True, stratify = validtest_labels)
            ret['test_idxs'] = cur_validtest[0]
            ret['valid_idxs'] = cur_validtest[1]
        else:
            ret['test_idxs'] = cur_train_validtest[1]
            ret['valid_idxs'] = np.array([])
    return ret





def get_train_test_subsets(dataset_obj, datadict, train_folds = UC.TRAIN_FOLDS, valid_folds =UC.VALID_FOLDS, test_folds = UC.TEST_FOLDS, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = UC.SEED):
    idx_dict = {}
    if datadict['train_on_middle'] == True or len(train_folds) == 0:
        # if train_folds is empty or training on middle, randomize with given pct/subpct splits

        idx_dict = get_train_test_splits(datadict, train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    else:
        # default to given folds
        num_examples = datadict['num_examples']
        _idxs = np.arange(num_examples)
        temp_df = pl.DataFrame({'fold': datadict['df']['fold'], 'idxs': _idxs})
        idx_dict['train_idxs'] = temp_df.filter(pl.col('fold').is_in(train_folds))['idxs'].to_numpy()
        idx_dict['valid_idxs'] = temp_df.filter(pl.col('fold').is_in(valid_folds))['idxs'].to_numpy()
        idx_dict['test_idxs'] = temp_df.filter(pl.col('fold').is_in(test_folds))['idxs'].to_numpy()
    train_subset = TUD.Subset(dataset_obj, idx_dict['train_idxs'])
    valid_subset = None
    test_subset = None
    weights = np.array([])
    train_size = idx_dict['train_idxs'].shape[0]
    valid_size = idx_dict['valid_idxs'].shape[0]
    test_size = idx_dict['test_idxs'].shape[0]
    if datadict['is_balanced'] == False:
        cur_label = datadict['label']
        train_df = datadict['df'][idx_dict['train_idxs']]
        class_amounts = {k:v[0] for (k,v) in train_df[cur_label].value_counts().rows_by_key(cur_label).items()}
        amount_arr = np.array([class_amounts[k] for k in datadict['label_arr']]).flatten()
        inv_class_prop = np.sum(amount_arr)/amount_arr
        weights = inv_class_prop/np.max(inv_class_prop)
    if idx_dict['valid_idxs'].shape[0] > 0:
        valid_subset = TUD.Subset(dataset_obj, idx_dict['valid_idxs'])
    if idx_dict['test_idxs'].shape[0] > 0:
        test_subset = TUD.Subset(dataset_obj, idx_dict['test_idxs'])
    ret = {
            'weights': weights,
            'train_subset': train_subset,
            'valid_subset': valid_subset,
            'test_subset': test_subset,
            'train_idxs': idx_dict['train_idxs'],
            'valid_idxs': idx_dict['valid_idxs'],
            'test_idxs': idx_dict['test_idxs'],
            'train_size': train_size,
            'valid_size': valid_size,
            'test_size': test_size,
            'train_folds': train_folds,
            'valid_folds': valid_folds,
            'test_folds': test_folds
            }
    return ret


# input torch, output torch
def accumulate_vecs(cur_vecs, vec_to_add):
    if cur_vecs == None:
        return vec_to_add
    else:
        return torch.vstack((cur_vecs, vec_to_add))

# input torch, output numpy
# predictions are probability dists, convert to index
def accumulate_truths_preds(truths, truths_to_add, preds, preds_to_add, batch_idx, is_classification = False):
    new_truths = truths_to_add.detach().cpu().numpy().flatten()
    new_preds = None
    if is_classification == True:
        new_preds = torch.argmax(preds_to_add,axis=1).detach().cpu().numpy().flatten()
    else:
        # regression doesn't need argmax
        new_preds = preds_to_add.detach().cpu().numpy().flatten()

    # first time through, just return new truths and preds
    if batch_idx == 0:
        return new_truths, new_preds
    else:
        # The base of an array that owns its memory is None
        # (and want to own own memory, so deep copy if not)
        # (doesn't work if truths is None, first time around)
        if truths.base is None and preds.base is None:
            return np.hstack((truths,new_truths)), np.hstack((preds, new_preds))
        else:
            return np.hstack((copy.deepcopy(truths),new_truths)), np.hstack((copy.deepcopy(preds), new_preds))

def save_scaler_dict(scaler, configdict, layer_idx):
    suffix = configdict['suffix']
    other_str = f'l{layer_idx}'

    cur_type = None
    if configdict['is_64bit'] == True:
        cur_type = 'scaler64'
    else:
        cur_type = 'scaler32'
    save_path = UMN.get_save_path(cur_type, configdict, other=other_str, make_dir = True)
    torch.save(scaler, save_path)

def load_scaler_dict(scaler, configdict, layer_idx, device='cpu'):
    other_str = f'l{layer_idx}'

    cur_type = None
    if configdict['is_64bit'] == True:
        cur_type = 'scaler64'
    else:
        cur_type = 'scaler32'
    save_path = UMN.get_save_path(cur_type, configdict, other=other_str, make_dir = False)

    scaler.load_state_dict(torch.load(save_path, map_location=device, weights_only = False))

def save_model_dict(model_dict, configdict, layer_idx, trial_number):
    suffix = configdict['suffix']
    layer_str = f'l{layer_idx}'
    trial_str = f't{trial_number}'
    other_str = f'{layer_str}_{trial_str}_{suffix}'
    save_path = UMN.get_save_path('model', configdict, other=other_str, make_dir = True)
    torch.save(model_dict, save_path)

def load_model_dict(model, configdict, layer_idx, trial_number, device='cpu'):
    suffix = configdict['suffix']
    layer_str = f'l{layer_idx}'
    trial_str = f't{trial_number}'
    other_str = f'{layer_str}_{trial_str}_{suffix}'
    save_path = UMN.get_save_path('model', configdict, other=other_str, make_dir = False)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only = False))

def save_mean(cur_mean, configdict, layer_idx, is_train = True):
    suffix = configdict['suffix']
    split_str = 'nil'
    if is_train == True:
        split_str = 'train'
    else:
        split_str = 'test'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-mean'
    save_path = UMN.get_save_path('mean', configdict, other=other_str, make_dir = True)
    np.save(save_path, cur_mean.cpu().numpy())

def load_mean(configdict, layer_idx, is_train = True):
    split_str = 'nil'
    if is_train == True:
        split_str = 'train'
    else:
        split_str = 'test'

    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-mean'
    save_path = UMN.get_save_path('mean', configdict, other=other_str, make_dir = False)
    return np.load(save_path)


def save_std(cur_std, configdict, layer_idx, is_train = True):
    suffix = configdict['suffix']
    split_str = 'nil'
    if is_train == True:
        split_str = 'train'
    else:
        split_str = 'test'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-std'
    save_path = UMN.get_save_path('std', configdict, other=other_str, make_dir = True)
    np.save(save_path, cur_std.cpu().numpy())

def load_std(configdict, layer_idx, is_train = True):
    split_str = 'nil'
    if is_train == True:
        split_str = 'train'
    else:
        split_str = 'test'

    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-std'
    save_path = UMN.get_save_path('std', configdict, other=other_str, make_dir = False)
    return np.load(save_path)

def save_part_rto(cur_pr, configdict, layer_idx, is_train = True):
    suffix = configdict['suffix']
    split_str = 'nil'
    if is_train == True:
        split_str = 'train'
    else:
        split_str = 'test'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}'
    save_path = UMN.get_save_path('part_rto', configdict, other=other_str, make_dir = True)
    np.save(save_path, cur_pr.cpu().numpy())

def load_part_rto(configdict, layer_idx, is_train = True):
    split_str = 'nil'
    if is_train == True:
        split_str = 'train'
    else:
        split_str = 'test'

    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}'
    save_path = UMN.get_save_path('part_rto', configdict, other=other_str, make_dir = False)
    return np.load(save_path)

def log_scaler_epoch_mean_var(run_name, scalerdict):
    means = scalerdict['mean_vecs_epoch'].detach().cpu().numpy()
    variances = scalerdict['var_vecs_epoch'].detach().cpu().numpy() 
    scaler_path = UMN.by_projpath(UC.SCALERS_DOC_FOLDER, make_dir = True)
    out_path_means = os.path.join(scaler_path, f'{run_name}-means.npy')
    np.save(out_path_means, means, allow_pickle = True)
    out_path_vars = os.path.join(scaler_path, f'{run_name}-vars.npy')
    np.save(out_path_vars, variances, allow_pickle = True)

