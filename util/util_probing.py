import copy
import os

from . import util_main as UMN
from . import util_constants as UC

import numpy as np
import polars as pl
import torch, torch.utils.data as TUD
from sklearn.model_selection import train_test_split


def create_splits(datadict, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = 39):
    idxs = np.arange(datadict'[num_examples'])
    label = datadict['label']
    labels = datadict['df'][label].to_numpy()
    preq_all_idxs, testvalid_idxs = train_test_split(idxs, train_size = train_pct, random_state = seed, shuffle = True, stratify = labels)
    test_idxs, valid_idxs = train_test_split(testvalid_idxs, train_size = test_subpct, random_state = seed, shuffle = True, stratify = labels[testvalid_idxs])
    _preq_idxs = []
    prev_idxs = preq_all_idxs
    for i in range(UC.DATASET_PREQ_STEPS[datadict['dataset']]):
        train_idxs, encode_idxs = train_test_split(prev_idxs, train_size = 0.5, random_state = seed, shuffle = True, stratify = labels[prev_idxs])
        _preq_idxs.append(encode_idxs)
        prev_idxs = train_idxs
        if i == num_steps - 1:
            _preq_idxs.append(train_idxs)


    preq_all_idxs_size = preq_idxs_all.shape[0]
    preq_idxs = _preq_idxs[::-1] 
    preq_size = [x.shape[0] for x in preq_idxs]
    valid_size = valid_idxs.shape[0]
    test_size = test_idxs.shape[0]
    ret ={'preq_idxs': preq_idxs, 'preq_size': preq_size,
          'valid_idxs': valid_idxs, 'valid_size': valid_size,
          'test_idxs': test_idxs, 'test_size': test_size,
          'preq_all_idxs': preq_idxs_all, 'preq_all_idxs_size': preq_all_idxs_size
          }
    return ret


def create_subsets(dataset_obj, idx_dict):
    for idx_type in ['preq', 'valid', 'test', 'preq_all']:
        idx_str = f'{idx_type}_idxs'
        subset_str = f'{idx_type}_subset'
        if idx_type == 'preq':
            for i in range(1,len(idx_dict[idx_str])):
                train_subset = 


def get_train_test_subsets(dataset_obj, datadict, train_folds = UC.TRAIN_FOLDS, test_folds = UC.TEST_FOLDS):
    idx_dict = {}
    fold_col = 'fold'
    # default to given folds
    num_examples = datadict['num_examples']
    num_classes = datadict['num_classes'] 
    _idxs = np.arange(num_examples)
    temp_df = pl.DataFrame({fold_col: datadict['df'][fold_col], 'idxs': _idxs})
    online_folds = []
    t1log2k = None
    for train_fold_idx in range(1,len(train_folds)):
        cur_train_online_folds = train_folds[:train_fold_idx]
        cur_eval_online_folds = train_folds[train_fold_idx]
        train_online_idxs = temp_df.filter(pl.col(fold_col).is_in(cur_train_online_folds))['idxs'].to_numpy()
        eval_online_idxs = temp_df.filter(pl.col(fold_col).is_in(cur_eval_online_folds))['idxs'].to_numpy()
        train_online_num_folds = len(cur_train_online_folds)
        eval_online_num_folds = len(cur_eval_online_folds)
        train_online_num_idxs = cur_train_online_idxs.shape[0]
        eval_online_num_idxs = cur_eval_online_idxs.shape[0]
        train_online_subset = TUD.Subset(dataset_obj, train_online_idxs)
        eval_online_subset = TUD.Subset(dataset_obj, eval_online_idxs)
        cur_dict = {'train_idxs': train_online_idxs,
                    'eval_idxs': eval_online_idxs,
                    'train_num_folds': train_online_num_folds,
                    'eval_num_folds': eval_online_num_folds,
                    'train_num_idxs': train_online_num_idxs,
                    'eval_num_idxs': eval_online_num_idxs,
                    'train_subset': train_online_subset,
                    'eval_subset': eval_online_subset
                    }
        if train_fold_idx == 1:
            t1log2k = train_online_num_idxs * np.log2(num_classes) 
        online_folds.append(cur_dict)

    idx_dict['online_folds'] = online_folds
    idx_dict['t1log2k'] = t1log2k
    idx_dict['full_train_idxs'] = temp_df.filter(pl.col(fold_col).is_in(train_folds))['idxs'].to_numpy()
    idx_dict['full_train_folds'] = train_folds
    idx_dict['full_train_size'] = idx_dict['full_train_idxs'].shape[0]
    idx_dict['test_idxs'] = temp_df.filter(pl.col(fold_col).is_in(test_folds))['idxs'].to_numpy()
    idx_dict['test_size'] = idx_dict['test_idxs'].shape[0]
    idx_dict['test_folds'] = test_folds
    if datadict['is_balanced'] == False:
        cur_label = datadict['label']
        train_df = datadict['df'][idx_dict['full_train_idxs']]
        class_amounts = {k:v[0] for (k,v) in train_df[cur_label].value_counts().rows_by_key(cur_label).items()}
        amount_arr = np.array([class_amounts[k] for k in datadict['label_arr']]).flatten()
        inv_class_prop = np.sum(amount_arr)/amount_arr
        idx_dict['weights'] = inv_class_prop/np.max(inv_class_prop)
    if idx_dict['full_train_idxs'].shape[0] > 0:
        idx_dict['full_train_subset'] = TUD.Subset(dataset_obj, idx_dict['full_train_idxs'])
    if idx_dict['test_idxs'].shape[0] > 0:
        idx_dict['test_subset'] = TUD.Subset(dataset_obj, idx_dict['test_idxs'])
    return idx_dict


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

