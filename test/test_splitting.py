from util import util_main as UMN
from util import util_constants as UC
from util import util_probing as UP
import torch, torch.utils.data as TUD
import polars as pl
import numpy as np

class TestDataset(TUD.Dataset):
    def __init__(self, datadict, classdict):
        self.df = datadict['df']
        self.dataset = datadict['dataset']
        self.label = datadict['label']
        self.classdict = classdict
    def __len__(self):
        return self.df['name'].count()


    def __getitem__(self,idx):
        cur_row = self.df.row(idx, named=True)
        cur_name = cur_row['name']
        cur_truth = self.classdict[cur_row[self.label]]
        return cur_name, cur_truth

def check_unique_subset(cur_subset, subset_sz):
    for didx, data in enumerate(TUD.DataLoader(cur_subset, batch_size=subset_sz, shuffle = True)):
        cur_data, cur_labels = data
        np_data = np.array(cur_data)
        np_labels = np.array(cur_labels)

        data_unique, data_counts = np.unique(np_data,return_counts=True)
        print('subset_size', subset_sz, 'data_shape', np_data.shape, 'label_shape', np_labels.shape, 'unique_shape', data_unique.shape)


def debug(cur_df, subsetdict):
    for idx_type in ['preq', 'valid', 'test', 'preq_all']:
        #idx_str = f'{idx_type}_idxs'
        #subset_str = f'{idx_type}_subset'
        #size_str = f'{idx_type}_size'
        if idx_type == 'preq':
            for pidx, pdict in enumerate(subsetdict['preq']):
                cur_train = pdict['train_idxs']
                cur_tsz = pdict['train_size']
                cur_esz = pdict['encode_size']
                cur_encode = pdict['encode_idxs']
                cur_tsb = pdict['train_subset']
                cur_esb = pdict['encode_subset']
                train_entries = cur_df[cur_train]
                encode_entries = cur_df[cur_encode]
                print(f'any preq_{pidx} train dup', train_entries.is_duplicated().to_numpy().any())
                check_unique_subset(cur_tsb, cur_tsz)
                print(f'any preq_{pidx} encode dup', encode_entries.is_duplicated().to_numpy().any())
                check_unique_subset(cur_esb, cur_esz)

                

        else:
            idx_str = f'{idx_type}_idxs'
            subset_str = f'{idx_type}_subset'
            size_str = f'{idx_type}_size'
            cur_train = subsetdict[idx_str]
            train_entries = cur_df[cur_train]
            cur_sb = subsetdict[subset_str]
            cur_sz = subsetdict[size_str]
            print(f'any {idx_type} dup', train_entries.is_duplicated().to_numpy().any())
            check_unique_subset(cur_sb, cur_sz)


def create_dataset():
    digits = [chr(x) for x in range(48,58)]
    uppers = [chr(x) for x in range(65,91)]
    lowers = [chr(x) for x in range(97,123)]
    #lowerupper = [f'{x}{y}' for x in lowers for y in uppers]
    #upperlower = [f'{x}{y}' for x in uppers for y in lowers]
    lowerdigit = [f'{x}{y}' for x in lowers for y in digits]
    upperdigit = [f'{x}{y}' for x in uppers for y in digits]
    digitdigit = [f'{x}{y}' for x in digits for y in digits]
    num_ld = len(lowerdigit) # 260, indices 0-259
    num_ud = len(upperdigit) # 260 indices 260-519
    num_dd = len(digitdigit) # 100 indices 520-619
    labels = ['lowerdigit'] * num_ld + ['upperdigit'] * num_ud + ['digitdigit'] * num_dd
    classdict = {'lowerdigit': 0, 'upperdigit': 1, 'digitdigit': 2}
    data = {'name': lowerdigit + upperdigit + digitdigit,
            'type': labels}
    df = pl.DataFrame(data, schema=[('name', pl.String), ('type', pl.String)])
    return df, classdict, (num_ld + num_ud + num_dd)


def create_subsets(dataset_obj, datadict, train_pct = UC.TRAIN_PCT, test_subpct = UC.TEST_SUBPCT, seed = 39):
    idxdict = UP.get_split_idxs(datadict, train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    #print(idxdict)
    subsetdict = UP.create_subsets_from_splits(dataset_obj, idxdict)
    debug(datadict['df'], subsetdict)
    #print(subsetdict.keys())
    return subsetdict
           
def test():
    df, classdict, num_examples = create_dataset()
    datadict = {'df': df, 'label': 'type', 'num_examples': num_examples, 'dataset': 'test_2char'}
    dataset_obj = TestDataset(datadict,classdict)
    UC.DATASET_PREQ_STEPS['test_2char'] = 3
    create_subsets(dataset_obj, datadict)


test()
