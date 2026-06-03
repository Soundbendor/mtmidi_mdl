import copy
import os

from . import util_main as UMN
from . import util_constants as UC

import numpy as np
import polars as pl
import torch, torch.utils.data as TUD
from sklearn.model_selection import train_test_split

def build_config(parser_args, datadict):
    _config = {k:v for (k,v) in vars(parser_args).items()}
    model_shape = UMN.get_acts_shape(_config['model_size'])
    _config['learning_rate'] = UC.LEARNING_RATE
    _config['num_epochs'] = UC.NUM_EPOCHS
    _config['seed'] = UC.SEED
    _config['batch_size'] = UC.BATCH_SIZE
    _config['is_64bit'] = UC.IS_64BIT
    _config['patience_delta'] = UC.PATIENCE_DELTA
    _config['model_dim'] = model_shape[1]
    _config['model_num_layers'] = model_shape[0]
    _config['dataloader_shuffle'] = UC.DATALOADER_SHUFFLE
    if parser_args.expr_type == 'mlp':
        _config['probe_hidden_dims'] = UC.MLPPROBE_HIDDEN_DIMS
    elif parser_args.expr_type == 'linear':
        _config['probe_hidden_dims'] = []
    _config['early_stopping_check_interval'] = UC.EARLY_STOPPING_CHECK_INTERVAL
    _config['early_stopping_patience'] = UC.EARLY_STOPPING_PATIENCE

    _config['train_pct'] = UC.TRAIN_PCT
    _config['test_subpct'] = UC.TEST_SUBPCT

    _config['is_balanced'] = datadict['is_balanced']
    _config['use_weights'] = UC.USE_WEIGHTS
    return _config

def get_split_idxs(datadict, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = 39):
    idxs = np.arange(datadict['num_examples'])
    label = datadict['label']
    labels = datadict['df'][label].to_numpy()
    preq_all_idxs, testvalid_idxs = train_test_split(idxs, train_size = train_pct, random_state = seed, shuffle = True, stratify = labels)
    test_idxs, valid_idxs = train_test_split(testvalid_idxs, train_size = test_subpct, random_state = seed, shuffle = True, stratify = labels[testvalid_idxs])
    _preq_idxs = []
    prev_idxs = preq_all_idxs

    num_steps = UC.DATASET_PREQ_STEPS[datadict['dataset']]
    for i in range(num_steps):
        train_idxs, encode_idxs = train_test_split(prev_idxs, train_size = 0.5, random_state = seed, shuffle = True, stratify = labels[prev_idxs])
        _preq_idxs.append(encode_idxs)
        prev_idxs = train_idxs
        if i == num_steps - 1:
            _preq_idxs.append(train_idxs)


    preq_all_idxs_size = preq_all_idxs.shape[0]
    preq_idxs = _preq_idxs[::-1] 
    preq_size = [x.shape[0] for x in preq_idxs]
    valid_size = valid_idxs.shape[0]
    test_size = test_idxs.shape[0]
    ret ={'preq_idxs': preq_idxs, 'preq_size': preq_size,
          'valid_idxs': valid_idxs, 'valid_size': valid_size,
          'test_idxs': test_idxs, 'test_size': test_size,
          'preq_all_idxs': preq_all_idxs, 'preq_all_size': preq_all_idxs_size,
          'num_preq_steps': num_steps,
          'train_pct': train_pct,
          'test_subpct': test_subpct
          }
    return ret


def create_subsets_from_splits(dataset_obj, idx_dict):
    ret = {}
    for idx_type in ['preq', 'valid', 'test', 'preq_all']:
        idx_str = f'{idx_type}_idxs'
        subset_str = f'{idx_type}_subset'
        size_str = f'{idx_type}_size'
        if idx_type == 'preq':
            ret['preq'] = []
            for i in range(1,len(idx_dict[idx_str])):
                cur = {}
                if i == 1:
                    cur['train_idxs'] = idx_dict[idx_str][0]
                else:
                    cur['train_idxs'] = np.hstack(idx_dict[idx_str][:i])
                cur['train_size'] = cur['train_idxs'].shape[0]
                cur['encode_idxs'] = idx_dict[idx_str][i]
                cur['encode_size'] = cur['encode_idxs'].shape[0]
                cur['train_subset'] = TUD.Subset(dataset_obj, cur['train_idxs'])
                cur['encode_subset'] = TUD.Subset(dataset_obj, cur['encode_idxs'])
                ret['preq'].append(cur)
        else:
            ret[idx_str] = idx_dict[idx_str]
            ret[size_str] = idx_dict[size_str]
            ret[subset_str] = TUD.Subset(dataset_obj, ret[idx_str])
    ret['num_preq_steps'] = idx_dict['num_preq_steps']
    ret['train_pct'] = idx_dict['train_pct']
    ret['test_subpct'] = idx_dict['test_subpct']
    return ret

def create_subsets(dataset_obj, datadict, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = 39):
    idxdict = get_split_idxs(datadict, train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    subsetdict = create_subsets_from_splits(dataset_obj, idxdict)
    return subsetdict
                



def get_preq_valid_test_subsets(dataset_obj, datadict, configdict):
    # default to given folds
    train_pct = configdict['train_pct']
    test_subpct = configdict['test_subpct']
    num_examples = datadict['num_examples']
    seed = configdict['seed']
    num_classes = datadict['num_classes'] 
    subsetdict = create_subsets(dataset_obj, datadict, train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    ret_dict = {k:v for (k,v) in subsetdict.items()}
    ret_dict['t1logk'] = subsetdict['preq'][0]['train_size'] * np.log2(num_classes)
    if datadict['is_balanced'] == False:
        cur_label = datadict['label']
        train_df = datadict['df'][subsetdict['preq_all_idxs']]
        class_amounts = {k:v[0] for (k,v) in train_df[cur_label].value_counts().rows_by_key(cur_label).items()}
        amount_arr = np.array([class_amounts[k] for k in datadict['label_arr']]).flatten()
        inv_class_prop = np.sum(amount_arr)/amount_arr
        ret_dict['weights'] = inv_class_prop/np.max(inv_class_prop)
    return ret_dict


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

def save_mean(cur_mean, configdict, layer_idx):
    seed = configdict['seed']
    split_str = f'sd{seed}_train'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-mean'
    save_path = UMN.get_save_path('mean', configdict, other=other_str, make_dir = True)
    np.save(save_path, cur_mean.cpu().numpy())

def load_mean(configdict, layer_idx):
    seed = configdict['seed']
    split_str = f'sd{seed}_train'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-mean'
    save_path = UMN.get_save_path('mean', configdict, other=other_str, make_dir = False)
    return np.load(save_path)


def save_std(cur_std, configdict, layer_idx):
    seed = configdict['seed']
    split_str = f'sd{seed}_train'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-std'
    save_path = UMN.get_save_path('std', configdict, other=other_str, make_dir = True)
    np.save(save_path, cur_std.cpu().numpy())

def load_std(configdict, layer_idx):
    seed = configdict['seed']
    split_str = f'sd{seed}_train'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}-std'
    save_path = UMN.get_save_path('std', configdict, other=other_str, make_dir = False)
    return np.load(save_path)

def save_part_rto(cur_pr, configdict, layer_idx):
    seed = configdict['seed']
    split_str = f'sd{seed}_train'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}'
    save_path = UMN.get_save_path('part_rto', configdict, other=other_str, make_dir = True)
    np.save(save_path, cur_pr.cpu().numpy())

def load_part_rto(configdict, layer_idx):
    seed = configdict['seed']
    split_str = f'sd{seed}_train'
    layer_str = f'l{layer_idx}'
    other_str = f'{layer_str}_{split_str}'
    save_path = UMN.get_save_path('part_rto', configdict, other=other_str, make_dir = False)
    return np.load(save_path)


