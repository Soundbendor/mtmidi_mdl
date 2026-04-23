# duplicating sklearn's standardscaler https://github.com/scikit-learn/scikit-learn/blob/b24c328a30/sklearn/preprocessing/_data.py#L723
# originally coded for https://github.com/soundbendor/mtmidi

# paper cited by sklearn:
# Tony F. Chan, Gene H. Golub and Randall J. LeVeque
# Algorithms for Computing the Sample Variance: Analysis and Recommendations
# The American Statistician, Vol. 37, No. 3 (Aug., 1983), pp. 242-247
# Taylor & Francis, Ltd. on behalf of the American Statistical Association

from torch import nn
import torch
import os

class StandardScaler(nn.Module):
    def __init__(self,with_mean = True, with_std = True, use_64bit = True, dim=4096, use_constant_feature_mask = True, device = 'cpu'):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.use_64bit = use_64bit
        self.use_constant_feature_mask = use_constant_feature_mask
        self.device = device
        self.dim = dim
        if self.use_64bit == True:
            self.ftype = torch.float64
        else:
            self.ftype = torch.float32
        
        # used for bounds checking as per sklearn
        self.eps = torch.finfo(self.ftype).eps

        self.mean = nn.Parameter(torch.zeros(dim, dtype=self.ftype, device = device), requires_grad = False)
        self.var = nn.Parameter(torch.zeros(dim, dtype=self.ftype, device = device), requires_grad = False)
        self.scale = nn.Parameter(torch.ones(dim, dtype=self.ftype, device = device), requires_grad = False)
        self.num_samples = nn.Parameter(torch.tensor(0, dtype=torch.int64, device = device), requires_grad = False)
  

        # thresh from sklearn's _handle_zeros_in_scale in preprocessing _data.py
        self.zero_thresh = 10. * self.eps

    def reset(self):
       
        self.mean = nn.Parameter(torch.zeros(self.dim, dtype=self.ftype, device = self.device), requires_grad = False)
        self.var = nn.Parameter(torch.zeros(self.dim, dtype=self.ftype, device = self.device), requires_grad = False)
        self.scale = nn.Parameter(torch.ones(self.dim, dtype=self.ftype, device = self.device), requires_grad = False)
        self.num_samples = nn.Parameter(torch.tensor(0, dtype=torch.int64, device = self.device), requires_grad = False)

    def partial_fit(self, X):
        # (normed = batch scaled)
        ### defining vars
        cur_data = X.clone().detach().to(self.ftype).to(self.device)
        old_mean = self.mean.clone()
        old_var = self.var.clone()
        old_num_samples = self.num_samples.clone()

        ### mean M_t calculation
        old_sum = old_mean * self.num_samples
        cur_sum = torch.sum(cur_data,axis=0) # sum over batch
        cur_bs = cur_data.shape[0]
        
        # corrected bs = cur_bs - num_nan, don't need to do since shouldn't have nan
        new_num_samples = old_num_samples + cur_bs
        new_mean = (cur_sum + old_sum)/new_num_samples # (1.5a but normed)

        ### var R_t calculation
        cur_mean = cur_sum/cur_bs
        temp_1 = cur_data - cur_mean # mean centering cur_data
        corr = torch.sum(temp_1, axis=0) # sum over batch, unsquared mean ctr. cur_data
        temp_2 = torch.pow(temp_1, 2.)
        uncorr_cur_unnorm_var = torch.sum(temp_2, axis=0) # sum over batch, squared mean ctr. cur_data
        corr_cur_unnorm_var = uncorr_cur_unnorm_var - torch.pow(corr, 2.)/cur_bs # correction as per (1.7)
        last_unnorm_var = old_var * old_num_samples
        
        q = old_num_samples.to(self.ftype)/cur_bs
        # (q/new_num_samples = m/(n * (m + n)) in paper, 1/q = n/m in paper)
        
        new_unnorm_var = None
        if old_num_samples.item() != 0:
            new_unnorm_var = last_unnorm_var + corr_cur_unnorm_var + ( (q/new_num_samples) * torch.pow((old_sum/q) - cur_sum, 2. )) # (1.5b) 
        else:
            new_unnorm_var = corr_cur_unnorm_var
        new_var = new_unnorm_var/new_num_samples

        ### update step
        self.mean.data = new_mean
        self.var.data = new_var
        self.num_samples.data = new_num_samples

        new_scale = torch.sqrt(new_var)
        
        where_zero = None
        if self.use_constant_feature_mask == True:
            # duplicating sklearn's _is_constant_feature in preprocessing _data.py
            # detects when variance falls within error bounds defined by Table 1. 3 in paper
            # letting paper K = mean * sqrt(n_samp/unnormed_var) = mean / sqrt(var) (2.2)
            # sub mean^2 = K^2 unnormed_var/n_samp in to _is_constant_feature expression
            # (var <= n_samp * eps * var + (n_samp * mean * eps)**2 )
            # where eps = u (in paper) and substitute var = unnorm_var/n_samp = S/N
            # some reduction yields 1 <= Nu + N^2 K^2 u^2 where RHS is paper Table 1. 3 
            # (not sure why sklearn is not using corrected two-pass bounds)
            # since bounds in paper are given by |S - Sbar|/S, they are detecting when |S-Sbar|/S <= 1

            where_zero = self.var <= (self.num_samples * self.var * self.eps) + torch.pow(self.num_samples * self.mean * self.eps, 2.)
        else:
            # duplicating sklearn's _handle_zeros_in_scale in preprocessing _data.py
            # which prevents div by small numbers or zero by replacing with 1.
            where_zero  = torch.isclose(new_scale, torch.zeros_like(new_scale), atol=self.zero_thresh)

        new_scale[where_zero] = 1.
        self.scale.data = new_scale

        # to impl, functionality of safe_accumulator_op as used in _incremental_mean_and_var in utils/extmath.py

    def fit(self, X):
        self.reset()
        self.partial_fit(X)


    def transform(self, X):
        orig_type = X.dtype
        cur_data = X.clone().detach().to(self.ftype).to(self.device)
        if self.with_mean == True:
            cur_data -= self.mean
        if self.with_std == True:
            cur_data /= self.scale
        if orig_type != self.ftype:
            return cur_data.to(orig_type)
        else:
            return cur_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def partial_fit_transform(self, X):
        self.partial_fit(X)
        return self.transform(X)

    # adding: getters for monitoring
    def get_mean(self):
        return self.mean.data.detach()

    def get_var(self):
        return self.var.data.detach()

    def get_scale(self):
        return self.scale.data.detach()

