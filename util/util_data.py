import torch
import os
import polars as pl

from . import util_main as UMN
from . import util_constants as UC
from data_helpers import polyrhythms as POLY
from data_helpers import dynamics as DYN
from data_helpers import seventh_chords as CH7
from data_helpers import mode_mixture as MM
from data_helpers import secondary_dominants as SD

from data_helpers import time_signatures as TSG
from data_helpers import chords as CHD
from data_helpers import notes as NTS
from data_helpers import scales as SCL
from data_helpers import intervals as IVL
from data_helpers import simple_progressions as SPG

def get_df(dataset):
    fname = f'{dataset}-metadata.csv'
    csvpath = os.path.join(UMN.by_projpath('csv', make_dir = False), fname)
    cur_data = pl.read_csv(csvpath)
    return cur_data



def get_dataset_proportions(dataset_size, num_steps = UC.PREQ_STEPS, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT):
    powers_of_two = np.power(2, np.arange(1,num_steps+1))
    train_prop = powers_of_two/np.max(powers_of_two)
    actual_prop = train_pct * train_prop
    test_prop = (1. - train_pct) * test_subpct
    dev_prop = (1. - train_pct) * (1. - test_subpct)
    ret = {'prop': {'prequential': {'train': train_prop, 'actual': actual_prop},
                    'train': train_pct,
                    'test': test_prop,
                    'dev': dev_prop},
           'actual': {'prequential': dataset_size * actual_prop,
                      'train': dataset_size * train_pct,
                      'test': dataset_size * test_prop,
                      'dev': dataset_size * dev_prop}
           }
    return ret

def check_dataset_proportions(dataset_size, num_classes, num_steps = UC.PREQ_STEPS, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, num_class_thresh = UC.NUM_CLASS_THRESH):
    prop_dict = get_dataset_proportions(dataset_size, num_steps = num_steps, train_pct = train_pct, test_subpct = test_subpct)
    breaks_on = []
    for split in ['prequential', 'train', 'test', 'dev']:
        split_amt = prop_dict['actual'][split]
        cur_size = None
        if split == 'prequential':
            cur_size = split_amt.min()
        else:
            cur_size = split_amt
        if cur_size/num_classes < num_class_thresh:
            breaks_on.append(split)
    return {'below_class_thresh': breaks_on, 'props': prop_dict}


# classdict: go from classes to indices
# idxdict: go from indices to classes
def load_data_dict(dataset):
    num_classes = None
    classdict = None
    idxdict = None
    label = None
    is_balanced = True
    is_classification = dataset != 'tempos'
    label_arr = None
    cur_df = get_df(dataset)
    num_examples = len(cur_df)
    train_on_middle = False
    if dataset == 'polyrhythms':
        num_classes = POLY.num_poly
        classdict = POLY.polystr_to_idx
        idxdict = POLY.idx_to_polystr
        label_arr = POLY.class_arr 
        label = 'poly'
    elif dataset == 'dynamics':
        is_balanced = False
        num_classes = DYN.num_categories
        classdict = DYN.dyn_category_to_idx
        idxdict = DYN.dyn_idx_to_category
        label_arr = DYN.dyn_categories
        label = 'dyn_category'
    elif dataset == "seventh_chords":
        num_classes =  CH7.num_chords
        classdict = CH7.quality_to_idx
        idxdict = CH7.idx_to_quality
        label_arr = CH7.class_arr
        label = 'quality'
    elif dataset == 'mode_mixture':
        num_classes = MM.num_is_modemix
        classdict = MM.is_modemix_to_idx
        idxdict = MM.idx_to_is_modemix
        label_arr = MM.is_modemix_arr
        label = 'is_modemix'
    elif dataset == 'secondary_dominants':
        num_classes = SD.num_subtypes
        classdict = SD.sub_type_to_idx
        idxdict = SD.idx_to_sub_type
        label_arr = SD.sub_type_arr 
        label = 'sub_type'
    elif dataset == 'tempos':
        num_classes = float('inf') # regression
        classdict = {} # regression, no classes
        label_arr = [] # regression, no classes
        train_on_middle = True
        label = 'bpm'
    elif dataset == 'time_signatures':
        num_classes = TSG.num_timesig
        classdict = TSG.timesig_to_idx
        idxdict = TSG.idx_to_timesig
        label_arr = TSG.class_arr 
        label = 'time_signature'
    elif dataset == 'chords':
        num_classes = CHD.num_chords
        classdict = CHD.quality_to_idx
        idxdict = CHD.idx_to_quality
        label_arr = CHD.class_arr
        label = 'chord_type'
    elif dataset == 'notes':
        num_classes = NTS.num_pc
        classdict = NTS.pc_to_idx
        idxdict = NTS.idx_to_pc
        label_arr = NTS.class_arr
        label = 'root_note_pitch_class'
    elif dataset == 'scales':
        num_classes = SCL.num_modes
        classdict = SCL.mode_to_idx
        idxdict = SCL.idx_to_mode
        label_arr = SCL.class_arr
        label = 'mode'
    elif dataset == 'intervals':
        num_classes = IVL.num_intervals
        classdict = IVL.interval_to_idx
        idxdict = IVL.idx_to_interval
        label_arr = IVL.class_arr
        label = 'interval'
    elif dataset == 'simple_progressions':
        num_classes = SPG.num_progs
        classdict = SPG.prog_to_idx
        idxdict = SPG.idx_to_prog
        label_arr =  SPG.prog_arr
        label = 'orig_prog'

    #label_arr = cur_df.select([label]).to_numpy().flatten()

    check_class_props = check_dataset_proportions(num_examples, num_classes, num_steps = UC.PREQ_STEPS, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, num_class_thresh = UC.NUM_CLASS_THRESH)
    ret = {
            'dataset': dataset,
            'num_classes': num_classes,
            'num_examples': num_examples,
            'df': cur_df,
            'classdict': classdict,
            'idxdict': idxdict,
            'is_classification': is_classification,
            'label': label,
            'label_arr': label_arr,
            'is_balanced': is_balanced,
            'train_on_middle': train_on_middle,
            'below_class_thresh': check_class_props['below_class_thresh']
            }
    return ret

def get_memmap_at_idx(fname_base, fold_num, model_size, dataset, layer_idx, use_64bit=False, to_torch = True, other_projdir = '', device='cpu'):
    fname = f'{fname_base}.dat'
    emb_file = UMN.get_acts_file(model_size, dataset=dataset, fname=fname, write = False, use_64bit = use_64bit, use_shape = None, other_projdir = other_projdir, fold_num = fold_num)
    cur = emb_file[layer_idx,:].copy()
    if to_torch == True:
        cur = torch.from_numpy(cur).to(device)
    return cur



