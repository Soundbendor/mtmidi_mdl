import torch

def get_mu_i_old(pt1, other_pts, batch_size=64):
    ret = torch.zeros(2, dtype=pt1.dtype)
    ret[0] = torch.inf
    ret[1] = torch.inf
    num_other = other_pts.shape[0]
    cur_split  = 0
    last_split = 0
    comp = None
    if len(pt1.shape) < 2:
        comp = pt1.unsqueeze(0)
    else:
        comp = pt1
    
    step_size = batch_size
    if batch_size < 0:
        step_size = num_other
    while last_split < num_other:
        cur_split = last_split + step_size
        len_split = other_pts[last_split:cur_split].shape[0]
        cur_dists = torch.cdist(comp, other_pts[last_split:cur_split], p=2)
        for i in range(len_split):
            cur_dist = cur_dists[0][i]
            if cur_dist < ret[1]:
                is_smallest = cur_dist < ret[0]
                if is_smallest == True:
                    ret[1] = ret[0]
                    ret[0] = cur_dist
                else:
                    ret[1] = cur_dist
        last_split = cur_split
    mu = 0.
    if ret[0] > 0.:
        mu = (ret[1]/ret[0])
    return mu

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
def get_dist_idx(_i,_j, N):
    i = min(_i,_j)
    j = max(_i,_j)
    return N * i + j - ((i+2) * (i+1)) // 2

def get_mu_i(cur_idx, N, dist_vec,device='cpu'):
    ret = torch.zeros(2, dtype=dist_vec.dtype, device=device)
    ret[0] = torch.inf
    ret[1] = torch.inf
    for other_idx in range(N):
        if other_idx != cur_idx:
            dist_idx = get_dist_idx(cur_idx, other_idx, N)
            cur_dist = dist_vec[dist_idx]
            if cur_dist < ret[1]:
                is_smallest = cur_dist < ret[0]
                if is_smallest == True:
                    ret[1] = ret[0]
                    ret[0] = cur_dist
                else:
                    ret[1] = cur_dist
    mu = 0.
    if ret[0] > 0.:
        mu = (ret[1]/ret[0])
    return mu





def calc_twonn(datapts, batch_size = 64, unused_pct = 0.0, device='cpu'):
    batch_dim = 0
    num_pts = datapts.shape[0]
    idxs = torch.arange(num_pts, device=device)
    mus = torch.zeros(num_pts, dtype=datapts.dtype, device=device)
    dists = torch.nn.functional.pdist(datapts, p=2)
    for i in range(num_pts):
        #mus[i] = get_mu_i_old(datapts[i], datapts.index_select(batch_dim, idxs[idxs!= i]), batch_size = batch_size)
        mus[i] = get_mu_i(i, num_pts, dists)
    mu_idxsort = torch.argsort(mus)
    p_emp = torch.zeros(num_pts, dtype=mus.dtype,device=device)
    p_emp_unsort = (torch.arange(num_pts, dtype=p_emp.dtype,device=device))/num_pts
    # map i/N to to proper mu indices
    p_emp[mu_idxsort] = p_emp_unsort
    #log_mus = torch.log(mus.index_select(0, idxs[idxs != max_idx])).unsqueeze(1)
    #log_emp = -1. * torch.log(1. - p_emp.index_select(0, idxs[idxs != max_idx])).unsqueeze(1)
    #log_mus = torch.log(mus.index_select(0, idxs)).unsqueeze(1)
    #log_emp = -1. * torch.log(1. - p_emp.index_select(0, idxs)).unsqueeze(1)
    log_mus = None
    log_emp = None
    if unused_pct > 0.:
        use_idxs = int(num_pts * (1. - unused_pct))
        #log_mus = torch.log(mus.index_select(0, mu_idxsort[:use_idxs])).unsqueeze(1)
        #log_emp = -1. * torch.log(1. - p_emp.index_select(0, mu_idxsort[:use_idxs])).unsqueeze(1)
        log_mus = torch.log(mus.index_select(0, mu_idxsort[:use_idxs]))
        log_emp = -1. * torch.log(1. - p_emp.index_select(0, mu_idxsort[:use_idxs]))
    else:
        #log_mus = torch.log(mus).unsqueeze(1)
        #log_emp = -1. * torch.log(1. - p_emp).unsqueeze(1)
        log_mus = torch.log(mus)
        log_emp = -1. * torch.log(1. - p_emp)
    #xtx = log_mus.T @ log_mus
    #xty = log_mus.T @ log_emp
    xtx = torch.sum(torch.pow(log_mus, 2.))
    xty = torch.sum(torch.mul(log_mus, log_emp))
    # least squares solution
    slope = xty/xtx
    return slope








