import torch, torch.utils.data as TUD

import util.util_data as UD

class ProbeDataset(TUD.Dataset):
    def __init__(self, datadict, model_size, layer_idx=0, from_dir = '', to_torch = True, emit_name = False, device ='cpu'):
        self.df = datadict['df']
        self.dataset = datadict['dataset']
        self.classdict = datadict['classdict']
        self.label = datadict['label']
        self.layer_idx = layer_idx
        self.device = device
        self.model_size = model_size
        self.from_dir = from_dir
        self.to_torch = to_torch
        self.is_64bit = False
        self.emit_name = emit_name

    def __len__(self):
        return self.df['name'].count()

    def set_layer_idx(self,idx):
        self.layer_idx = idx

    def set_emit_name(self, emit_name):
        self.emit_name = emit_name

    def __getitem__(self,idx):
        cur_row = self.df.row(idx, named=True)
        cur_name = cur_row['name']
        cur_fold = cur_row['fold']
        cur_truth = self.classdict[cur_row[self.label]]
        cur_arr = UD.get_memmap_at_idx(f'{cur_name}', cur_fold, self.model_size, self.dataset, self.layer_idx, use_64bit=self.is_64bit, to_torch = self.to_torch, other_projdir = self.from_dir, device=self.device)
        if self.emit_name == False:
            return cur_arr, cur_truth
        else:
            return cur_arr, cur_truth, cur_name



