import torch

def get_mu_i(pt1, other_pts, batch_size=64):
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
    while last_split <= other_pts.shape[0]:
        cur_split = last_split + batch_size
        len_split = other_pts[last_split:cur_split].shape[0]
        cur_dists = torch.cdist(comp, other_pts[last_split:cur_split], p=2)
        for i in range(len_split):
            cur_dist = cur_dists[0][i].item()
            if cur_dist < ret[1].item():
                is_smallest = cur_dist < ret[0].item()
                if is_smallest == True:
                    ret[1] = ret[0].item()
                    ret[0] = cur_dist
                else:
                    ret[1] = cur_dist
        last_split = cur_split
    mu = 0.
    if ret[0].item() > 0.:
        mu = (ret[1]/ret[0]).item()
    return mu

def calc_twonn(datapts, batch_size = 64, unused_pct = 0.0):
    batch_dim = 0
    num_pts = datapts.shape[0]
    idxs = torch.arange(num_pts)
    mus = torch.zeros(num_pts, dtype=datapts.dtype)
    for i in range(num_pts):
        mus[i] = get_mu_i(datapts[i], datapts.index_select(batch_dim, idxs[idxs!= i]), batch_size = batch_size)
    mu_idxsort = torch.argsort(mus)
    p_emp = torch.zeros(num_pts, dtype=mus.dtype)
    p_emp_unsort = (torch.arange(num_pts, dtype=p_emp.dtype))/num_pts
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
        log_mus = torch.log(mus.index_select(0, mu_idxsort[:use_idxs])).unsqueeze(1)
        log_emp = -1. * torch.log(1. - p_emp.index_select(0, mu_idxsort[:use_idxs])).unsqueeze(1)
    else:
        log_mus = torch.log(mus).unsqueeze(1)
        log_emp = -1. * torch.log(1. - p_emp).unsqueeze(1)
    xtx = log_mus.T @ log_mus
    xty = log_mus.T @ log_emp
    # least squares solution
    slope = xty/xtx
    return slope.item()








