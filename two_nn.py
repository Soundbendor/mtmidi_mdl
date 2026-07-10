import torch

def get_dist_idx(_i,_j, N):
    i = min(_i,_j)
    j = max(_i,_j)
    return N * i + j - ((i+2) * (i+1)) // 2

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
def get_dist_idxs(cur_idx, idxs, N, device='cpu'):
    ipts = torch.vstack((torch.ones(N-1, dtype=torch.int64, device=device) * cur_idx, torch.hstack((idxs[:cur_idx], idxs[(cur_idx+1):])))).sort(axis=0,descending=False).values
    #ipts = torch.vstack((torch.tensor([cur_idx], dtype=torch.int64, device=device).repeat(N-1), torch.hstack((idxs[:cur_idx], idxs[(cur_idx+1):])))).sort(axis=0,descending=False).values
    ret = N * ipts[0] + ipts[1] - ((ipts[0]+2) * (ipts[0]+1)) // 2
    return ret


def get_mu_i(cur_idx, idxs, N, dist_vec,device='cpu'):
    #log_mus = torch.log(mus.index_select(0, idxs[idxs != max_idx])).unsqueeze(1)
    dist_idxs = get_dist_idxs(cur_idx, idxs, N)
    cur_dists = dist_vec[dist_idxs]
    ret = cur_dists.sort(axis=0).values[:2]
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
        mus[i] = get_mu_i(i, idxs, num_pts, dists)
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

